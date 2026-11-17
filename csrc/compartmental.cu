#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace {
constexpr int threads = 256;

struct Matrix {
  const float* data;
  int64_t batch_stride, neuron_stride;
  float scalar;
  __device__ float get(int64_t b, int64_t n) const {
    return data ? data[b * batch_stride + n * neuron_stride] : scalar;
  }
};

struct Parameters {
  const float* data[6];
  int64_t stride[6];
  __device__ float get(int p, int64_t n) const {
    return data[p][n * stride[p]];
  }
};

void check_float(const at::Tensor& value, const at::Tensor& state) {
  TORCH_CHECK(value.is_cuda() && value.device() == state.device(),
              "all tensors must be on the state's CUDA device");
  TORCH_CHECK(value.scalar_type() == at::kFloat, "all values must be float32");
}

Matrix matrix(const at::Tensor& value, const at::Tensor& state) {
  check_float(value, state);
  TORCH_CHECK(value.sizes() == state.sizes(), "state/current/drive shape mismatch");
  return {value.data_ptr<float>(), value.stride(0), value.stride(1), 0.0f};
}

Matrix drive(const std::optional<at::Tensor>& value, double scalar,
             const at::Tensor& state) {
  if (value.has_value()) return matrix(*value, state);
  return {nullptr, 0, 0, static_cast<float>(scalar)};
}

Parameters parameters(const std::vector<at::Tensor>& values, const at::Tensor& state) {
  TORCH_CHECK(values.size() == 6, "six neuron parameter vectors are required");
  Parameters result;
  for (int i = 0; i < 6; ++i) {
    check_float(values[i], state);
    TORCH_CHECK(values[i].dim() == 1 && values[i].numel() == state.size(1),
                "parameters must have one value per neuron");
    result.data[i] = values[i].data_ptr<float>();
    result.stride[i] = values[i].stride(0);
  }
  return result;
}

void check_state(const at::Tensor& state) {
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat,
              "state must be CUDA float32");
  TORCH_CHECK(state.dim() == 2 && state.size(1) > 0, "state must be [B,N], N > 0");
}

void check_index(const at::Tensor& index, const at::Tensor& state) {
  TORCH_CHECK(index.device() == state.device() && index.scalar_type() == at::kInt
              && index.is_contiguous() && index.dim() == 1,
              "indices must be contiguous int32 vectors on the state device");
}

// One calculation shared by matrix and event kernels. All reads precede writes.
__device__ void advance(float current, int64_t row, int64_t n, int64_t count,
                        int64_t neurons, Matrix s, Matrix d, Matrix a,
                        Matrix input, Matrix dendritic_input, Parameters p,
                        float* output) {
  const int64_t b = row / neurons;
  const float alpha = p.get(0, n), gain = p.get(1, n);
  const float rd = p.get(2, n), ra = p.get(3, n);
  const float coupling = p.get(4, n), strength = p.get(5, n);
  const float dendrite = rd * d.get(b, n) +
      (1.0f - rd) * tanhf(current + dendritic_input.get(b, n));
  const float soma_input = (1.0f - coupling) * current + coupling * dendrite
      - strength * a.get(b, n) + input.get(b, n);
  const float next = fminf(1.0f, fmaxf(0.0f,
      alpha * s.get(b, n) + (1.0f - alpha) * gain * tanhf(soma_input)));
  output[row] = next;
  output[count + row] = dendrite;
  output[2 * count + row] = ra * a.get(b, n) + (1.0f - ra) * next;
}

template<int width>
__global__ void matrix_kernel(Matrix s, Matrix d, Matrix a, Matrix input,
                               Matrix dendritic_input, Parameters p,
                               const int32_t* rowptr, const int32_t* source,
                               const float* weights, float* output,
                               int64_t neurons, int64_t count) {
  const int lane = threadIdx.x % width;
  for (int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / width;
       row < count; row += static_cast<int64_t>(gridDim.x) * blockDim.x / width) {
    const int64_t b = row / neurons, n = row % neurons;
    float current = 0;
    for (int64_t e = static_cast<int64_t>(rowptr[n]) + lane; e < rowptr[n + 1]; e += width)
      current = fmaf(weights[e], s.get(b, source[e]), current);
    const unsigned mask = __activemask();
    for (int offset = width / 2; offset; offset /= 2)
      current += __shfl_down_sync(mask, current, offset, width);
    if (lane == 0)
      advance(current, row, n, count, neurons, s, d, a, input, dendritic_input, p, output);
  }
}

__global__ void event_advance_kernel(Matrix s, Matrix d, Matrix a, Matrix current,
                                     Matrix emitted, Matrix input, Matrix dendritic_input,
                                     Parameters p, const float* theta, int64_t theta_stride,
                                     float* output, float* delta,
                                     int64_t neurons, int64_t count) {
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < count; row += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = row / neurons, n = row % neurons;
    advance(current.get(b, n), row, n, count, neurons, s, d, a,
            input, dendritic_input, p, output);
    const float previous = emitted.get(b, n);
    const float residual = output[row] - previous;
    const bool active = fabsf(residual) >= theta[n * theta_stride];
    delta[row] = active ? residual : 0.0f;
    output[3 * count + row] = active ? output[row] : previous;
  }
}

__global__ void scatter_kernel(const float* delta, float* current,
                               const int32_t* rowptr, const int32_t* dst,
                               const int32_t* weight_index, const float* weights,
                               int64_t neurons, int64_t count) {
  const int lane = threadIdx.x % 32;
  for (int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
       row < count; row += static_cast<int64_t>(gridDim.x) * blockDim.x / 32) {
    const float value = delta[row];
    if (value == 0.0f) continue;
    const int64_t b = row / neurons, n = row % neurons;
    for (int64_t e = static_cast<int64_t>(rowptr[n]) + lane; e < rowptr[n + 1]; e += 32)
      atomicAdd(current + b * neurons + dst[e], value * weights[weight_index[e]]);
  }
}
}  // namespace

std::vector<at::Tensor> dendritic_step_cuda(
    at::Tensor state, at::Tensor dendrite, at::Tensor adaptation,
    std::vector<at::Tensor> params, at::Tensor rowptr, at::Tensor source,
    at::Tensor weight, std::optional<at::Tensor> input, double input_scalar,
    std::optional<at::Tensor> dendritic_input, double dendritic_scalar, int64_t width) {
  check_state(state);
  const c10::cuda::CUDAGuard guard(state.device());
  const auto s = matrix(state, state), d = matrix(dendrite, state), a = matrix(adaptation, state);
  const auto p = parameters(params, state);
  const auto u = drive(input, input_scalar, state), v = drive(dendritic_input, dendritic_scalar, state);
  check_index(rowptr, state); check_index(source, state); check_float(weight, state);
  TORCH_CHECK(rowptr.numel() == state.size(1) + 1 && weight.numel() == source.numel()
              && weight.is_contiguous() && weight.dim() == 1, "invalid CSR geometry");
  TORCH_CHECK(width == 8 || width == 16 || width == 32, "width must be 8, 16 or 32");
  auto output = at::empty({3, state.size(0), state.size(1)}, state.options());
  const int64_t count = state.numel();
  if (count) {
    const int blocks = static_cast<int>(std::min<int64_t>(4096, (count * width + threads - 1) / threads));
    const auto stream = c10::cuda::getCurrentCUDAStream(state.get_device());
#define LAUNCH(W) matrix_kernel<W><<<blocks, threads, 0, stream>>>(s,d,a,u,v,p, rowptr.data_ptr<int32_t>(),source.data_ptr<int32_t>(),weight.data_ptr<float>(),output.data_ptr<float>(),state.size(1),count)
    if (width == 8) { LAUNCH(8); }
    else if (width == 16) { LAUNCH(16); }
    else { LAUNCH(32); }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output.select(0, 0), output.select(0, 1), output.select(0, 2)};
}

std::vector<at::Tensor> dendritic_event_cuda(
    at::Tensor state, at::Tensor dendrite, at::Tensor adaptation,
    std::vector<at::Tensor> params, at::Tensor current, at::Tensor emitted,
    at::Tensor theta, at::Tensor rowptr, at::Tensor dst, at::Tensor weight_index,
    at::Tensor weight, std::optional<at::Tensor> input, double input_scalar,
    std::optional<at::Tensor> dendritic_input, double dendritic_scalar) {
  check_state(state);
  const c10::cuda::CUDAGuard guard(state.device());
  const auto s = matrix(state, state), d = matrix(dendrite, state), a = matrix(adaptation, state);
  const auto j = matrix(current, state), e = matrix(emitted, state);
  const auto p = parameters(params, state);
  const auto u = drive(input, input_scalar, state), v = drive(dendritic_input, dendritic_scalar, state);
  check_index(rowptr, state); check_index(dst, state); check_index(weight_index, state);
  check_float(weight, state); check_float(theta, state);
  TORCH_CHECK(current.is_contiguous(), "persistent current must be contiguous");
  TORCH_CHECK(rowptr.numel() == state.size(1) + 1 && dst.numel() == weight.numel()
              && dst.numel() == weight_index.numel() && weight.is_contiguous()
              && weight.dim() == 1, "invalid event topology geometry");
  TORCH_CHECK(theta.dim() == 1 && theta.numel() == state.size(1), "theta must have N values");
  auto output = at::empty({4, state.size(0), state.size(1)}, state.options());
  auto delta = at::empty({state.size(0), state.size(1)}, state.options());
  const int64_t count = state.numel();
  if (count) {
    const auto stream = c10::cuda::getCurrentCUDAStream(state.get_device());
    const int blocks = static_cast<int>(std::min<int64_t>(4096, (count + threads - 1) / threads));
    event_advance_kernel<<<blocks, threads, 0, stream>>>(s,d,a,j,e,u,v,p,
        theta.data_ptr<float>(),theta.stride(0),output.data_ptr<float>(),delta.data_ptr<float>(),state.size(1),count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int scatter_blocks = static_cast<int>(std::min<int64_t>(4096, (count * 32 + threads - 1) / threads));
    scatter_kernel<<<scatter_blocks, threads, 0, stream>>>(delta.data_ptr<float>(),current.data_ptr<float>(),
        rowptr.data_ptr<int32_t>(),dst.data_ptr<int32_t>(),weight_index.data_ptr<int32_t>(),
        weight.data_ptr<float>(),state.size(1),count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output.select(0,0),output.select(0,1),output.select(0,2),output.select(0,3)};
}
