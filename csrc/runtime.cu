#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace {

constexpr int kThreads = 256;

void check_float_cuda(
    const at::Tensor& tensor,
    const char* name,
    bool require_contiguous = true) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      !require_contiguous || tensor.is_contiguous(),
      name,
      " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
}

void check_cuda(cudaError_t error, const char* operation) {
  TORCH_CHECK(
      error == cudaSuccess,
      operation,
      " failed: ",
      cudaGetErrorString(error));
}

__global__ void advance_kernel(
    const float* __restrict__ state,
    const float* __restrict__ current,
    const float* __restrict__ alpha,
    const float* __restrict__ gain,
    float* __restrict__ output,
    int64_t count,
    int64_t neurons,
    int64_t current_batch_stride,
    int64_t current_neuron_stride,
    float drive) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t neuron = index % neurons;
    const int64_t batch = index / neurons;
    const float a = alpha[neuron];
    const int64_t current_index =
        batch * current_batch_stride + neuron * current_neuron_stride;
    const float equilibrium = gain[neuron] * tanhf(current[current_index] + drive);
    output[index] = fminf(1.0f, fmaxf(0.0f, fmaf(a, state[index], (1.0f - a) * equilibrium)));
  }
}

__global__ void advance_train_forward_kernel(
    const float* __restrict__ state,
    const float* __restrict__ current,
    const float* __restrict__ alpha,
    const float* __restrict__ gain,
    float* __restrict__ output,
    float* __restrict__ state_derivative,
    float* __restrict__ current_derivative,
    int64_t count,
    int64_t neurons,
    int64_t current_batch_stride,
    int64_t current_neuron_stride,
    float drive) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t neuron = index % neurons;
    const int64_t batch = index / neurons;
    const int64_t current_index =
        batch * current_batch_stride + neuron * current_neuron_stride;
    const float a = alpha[neuron];
    const float tanh_value = tanhf(current[current_index] + drive);
    const float beta_gain = (1.0f - a) * gain[neuron];
    const float preclip = fmaf(a, state[index], beta_gain * tanh_value);
    const float active = (preclip > 0.0f && preclip < 1.0f) ? 1.0f : 0.0f;
    output[index] = fminf(1.0f, fmaxf(0.0f, preclip));
    state_derivative[index] = active * a;
    current_derivative[index] =
        active * beta_gain * (1.0f - tanh_value * tanh_value);
  }
}

__global__ void advance_train_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ state_derivative,
    const float* __restrict__ current_derivative,
    float* __restrict__ grad_state,
    float* __restrict__ grad_current,
    int64_t count) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float gradient = grad_output[index];
    grad_state[index] = gradient * state_derivative[index];
    grad_current[index] = gradient * current_derivative[index];
  }
}

__global__ void update_neuron_traces_kernel(
    float* __restrict__ pre_trace,
    float* __restrict__ post_trace,
    const float* __restrict__ previous_state,
    const float* __restrict__ next_state,
    int64_t count,
    float pre_decay,
    float post_decay) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    pre_trace[index] = fmaf(
        pre_decay, pre_trace[index], (1.0f - pre_decay) * previous_state[index]);
    post_trace[index] = fmaf(
        post_decay, post_trace[index], (1.0f - post_decay) * next_state[index]);
  }
}

__global__ void update_edge_eligibility_kernel(
    const float* __restrict__ pre_trace,
    const float* __restrict__ post_trace,
    float* __restrict__ eligibility,
    const float* __restrict__ previous_state,
    const float* __restrict__ next_state,
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ dst,
    int64_t edges,
    int64_t neurons,
    int64_t batch,
    float eligibility_decay,
    float potentiation_scale,
    float depression_scale) {
  for (int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
       edge < edges;
       edge += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t source = src[edge];
    const int64_t target = dst[edge];
    float local = 0.0f;
    for (int64_t item = 0; item < batch; ++item) {
      const int64_t source_index = item * neurons + source;
      const int64_t target_index = item * neurons + target;
      local += potentiation_scale * pre_trace[source_index] * next_state[target_index]
          - depression_scale * previous_state[source_index] * post_trace[target_index];
    }
    local /= static_cast<float>(batch);
    local = fminf(1.0f, fmaxf(-1.0f, local));
    const float updated = fmaf(
        eligibility_decay,
        eligibility[edge],
        (1.0f - eligibility_decay) * local);
    eligibility[edge] = fminf(1.0f, fmaxf(-1.0f, updated));
  }
}

__global__ void dopamine_weight_update_kernel(
    float* __restrict__ source_weight,
    float* __restrict__ destination_weight,
    const int32_t* __restrict__ out_weight_index,
    const float* __restrict__ eligibility,
    const float* __restrict__ phasic_signal,
    const int32_t* __restrict__ dst,
    int64_t edges,
    float learning_rate,
    float weight_decay,
    float minimum_weight,
    float maximum_weight,
    bool preserve_sign,
    bool update_magnitude,
    int* __restrict__ maximum_delta_bits) {
  for (int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
       edge < edges;
       edge += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float previous = source_weight[edge];
    float direction = 1.0f;
    if (update_magnitude) {
      direction = previous > 0.0f ? 1.0f : (previous < 0.0f ? -1.0f : 0.0f);
    }
    float candidate = previous * (1.0f - weight_decay) +
        learning_rate * direction * phasic_signal[dst[edge]] * eligibility[edge];
    float lower = minimum_weight;
    float upper = maximum_weight;
    if (preserve_sign) {
      if (previous > 0.0f) {
        lower = fmaxf(lower, 0.0f);
      } else if (previous < 0.0f) {
        upper = fminf(upper, 0.0f);
      } else {
        lower = 0.0f;
        upper = 0.0f;
      }
    }
    const float updated = fminf(upper, fmaxf(lower, candidate));
    source_weight[edge] = updated;
    destination_weight[out_weight_index[edge]] = updated;
    atomicMax(maximum_delta_bits, __float_as_int(fabsf(updated - previous)));
  }
}

__global__ void advance_compact_kernel(
    float* __restrict__ state,
    const float* __restrict__ current,
    float* __restrict__ emitted,
    const float* __restrict__ alpha,
    const float* __restrict__ gain,
    const float* __restrict__ theta,
    int64_t* __restrict__ active_index,
    float* __restrict__ active_delta,
    int* __restrict__ active_count,
    int64_t count,
    int64_t neurons,
    float drive) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t neuron = index % neurons;
    const float a = alpha[neuron];
    const float equilibrium = gain[neuron] * tanhf(current[index] + drive);
    const float next = fminf(
        1.0f,
        fmaxf(0.0f, fmaf(a, state[index], (1.0f - a) * equilibrium)));
    state[index] = next;
    const float delta = next - emitted[index];
    if (fabsf(delta) >= theta[neuron]) {
      const int slot = atomicAdd(active_count, 1);
      active_index[slot] = index;
      active_delta[slot] = delta;
      emitted[index] = next;
    }
  }
}

__global__ void advance_compact_modulated_kernel(
    const float* __restrict__ state,
    const float* __restrict__ current,
    float* __restrict__ emitted,
    const float* __restrict__ alpha,
    const float* __restrict__ gain,
    const float* __restrict__ drive_shift,
    const float* __restrict__ gain_scale,
    const float* __restrict__ theta,
    float* __restrict__ output,
    int64_t* __restrict__ active_index,
    float* __restrict__ active_delta,
    int* __restrict__ active_count,
    int64_t count,
    int64_t neurons,
    float drive) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t neuron = index % neurons;
    const float a = alpha[neuron];
    const float equilibrium = gain[neuron] * gain_scale[index] *
        tanhf(current[index] + drive + drive_shift[index]);
    const float next = fminf(
        1.0f,
        fmaxf(0.0f, fmaf(a, state[index], (1.0f - a) * equilibrium)));
    output[index] = next;
    const float delta = next - emitted[index];
    if (fabsf(delta) >= theta[neuron]) {
      const int slot = atomicAdd(active_count, 1);
      active_index[slot] = index;
      active_delta[slot] = delta;
      emitted[index] = next;
    }
  }
}

__global__ void scatter_active_edges_kernel(
    float* __restrict__ current,
    const int32_t* __restrict__ out_rowptr,
    const int32_t* __restrict__ dst,
    const int32_t* __restrict__ out_weight_index,
    const float* __restrict__ weight,
    const int64_t* __restrict__ active_index,
    const float* __restrict__ active_delta,
    const int* __restrict__ active_count,
    int64_t neurons) {
  const int count = *active_count;
  constexpr int kWarpSize = 32;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int first_active = blockIdx.x * warps_per_block + warp_in_block;
  const int active_stride = gridDim.x * warps_per_block;
  for (int active = first_active; active < count; active += active_stride) {
    const int64_t flat_source = active_index[active];
    const int64_t batch = flat_source / neurons;
    const int64_t source = flat_source - batch * neurons;
    const float delta = active_delta[active];
    const int64_t begin = out_rowptr[source];
    const int64_t end = out_rowptr[source + 1];
    for (int64_t edge = begin + lane; edge < end; edge += kWarpSize) {
      const int64_t target = batch * neurons + dst[edge];
      atomicAdd(current + target, weight[out_weight_index[edge]] * delta);
    }
  }
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void canonical_spmm_kernel(
    const float* __restrict__ state,
    const int32_t* __restrict__ in_rowptr,
    const int32_t* __restrict__ in_src,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t neurons,
    int64_t rows) {
  constexpr int kWarpSize = 32;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t first_row =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t row_stride =
      static_cast<int64_t>(gridDim.x) * warps_per_block;
  for (int64_t row = first_row; row < rows; row += row_stride) {
    const int64_t batch = row / neurons;
    const int64_t target = row - batch * neurons;
    const int64_t begin = in_rowptr[target];
    const int64_t end = in_rowptr[target + 1];
    float sum = 0.0f;
    for (int64_t position = begin + lane; position < end; position += kWarpSize) {
      const int64_t source = in_src[position];
      sum = fmaf(weight[position], state[batch * neurons + source], sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
      output[row] = sum;
    }
  }
}

__global__ void canonical_step_kernel(
    const float* __restrict__ state,
    const int32_t* __restrict__ in_rowptr,
    const int32_t* __restrict__ in_src,
    const float* __restrict__ weight,
    const float* __restrict__ alpha,
    const float* __restrict__ gain,
    float* __restrict__ output,
    int64_t neurons,
    int64_t rows,
    float drive) {
  constexpr int kWarpSize = 32;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t first_row =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t row_stride =
      static_cast<int64_t>(gridDim.x) * warps_per_block;
  for (int64_t row = first_row; row < rows; row += row_stride) {
    const int64_t batch = row / neurons;
    const int64_t target = row - batch * neurons;
    const int64_t begin = in_rowptr[target];
    const int64_t end = in_rowptr[target + 1];
    float current = 0.0f;
    for (int64_t position = begin + lane; position < end; position += kWarpSize) {
      const int64_t source = in_src[position];
      current = fmaf(weight[position], state[batch * neurons + source], current);
    }
    current = warp_sum(current);
    if (lane == 0) {
      const float a = alpha[target];
      const float equilibrium = gain[target] * tanhf(current + drive);
      output[row] = fminf(
          1.0f,
          fmaxf(0.0f, fmaf(a, state[row], (1.0f - a) * equilibrium)));
    }
  }
}

__global__ void canonical_modulated_step_kernel(
    const float* __restrict__ state,
    const int32_t* __restrict__ in_rowptr,
    const int32_t* __restrict__ in_src,
    const float* __restrict__ weight,
    const float* __restrict__ alpha,
    const float* __restrict__ gain,
    const float* __restrict__ drive_shift,
    const float* __restrict__ gain_scale,
    float* __restrict__ output,
    int64_t neurons,
    int64_t rows,
    float drive) {
  constexpr int kWarpSize = 32;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t first_row =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t row_stride =
      static_cast<int64_t>(gridDim.x) * warps_per_block;
  for (int64_t row = first_row; row < rows; row += row_stride) {
    const int64_t batch = row / neurons;
    const int64_t target = row - batch * neurons;
    const int64_t begin = in_rowptr[target];
    const int64_t end = in_rowptr[target + 1];
    float current = 0.0f;
    for (int64_t position = begin + lane; position < end; position += kWarpSize) {
      const int64_t source = in_src[position];
      current = fmaf(weight[position], state[batch * neurons + source], current);
    }
    current = warp_sum(current);
    if (lane == 0) {
      const float a = alpha[target];
      const float equilibrium = gain[target] * gain_scale[row] *
          tanhf(current + drive + drive_shift[row]);
      output[row] = fminf(
          1.0f,
          fmaxf(0.0f, fmaf(a, state[row], (1.0f - a) * equilibrium)));
    }
  }
}

}  // namespace

at::Tensor advance_cuda(
    at::Tensor state,
    at::Tensor current,
    at::Tensor alpha,
    at::Tensor gain,
    double drive,
    int64_t stream_handle) {
  check_float_cuda(state, "state");
  check_float_cuda(current, "current", false);
  check_float_cuda(alpha, "alpha");
  check_float_cuda(gain, "gain");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, neurons]");
  TORCH_CHECK(current.sizes() == state.sizes(), "current shape must match state");
  TORCH_CHECK(alpha.numel() == state.size(1), "alpha length must equal neurons");
  TORCH_CHECK(gain.numel() == state.size(1), "gain length must equal neurons");

  check_cuda(cudaSetDevice(state.get_device()), "cudaSetDevice");
  auto output = at::empty_like(state);
  const int64_t count = state.numel();
  const int blocks = static_cast<int>(std::min<int64_t>((count + kThreads - 1) / kThreads, 4096));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  advance_kernel<<<blocks, kThreads, 0, stream>>>(
      state.data_ptr<float>(),
      current.data_ptr<float>(),
      alpha.data_ptr<float>(),
      gain.data_ptr<float>(),
      output.data_ptr<float>(),
      count,
      state.size(1),
      current.stride(0),
      current.stride(1),
      static_cast<float>(drive));
  check_cuda(cudaGetLastError(), "advance kernel launch");
  return output;
}

std::vector<at::Tensor> advance_train_forward_cuda(
    at::Tensor state,
    at::Tensor current,
    at::Tensor alpha,
    at::Tensor gain,
    double drive,
    int64_t stream_handle) {
  check_float_cuda(state, "state");
  check_float_cuda(current, "current", false);
  check_float_cuda(alpha, "alpha");
  check_float_cuda(gain, "gain");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, neurons]");
  TORCH_CHECK(current.sizes() == state.sizes(), "current shape must match state");
  TORCH_CHECK(alpha.numel() == state.size(1), "alpha length must equal neurons");
  TORCH_CHECK(gain.numel() == state.size(1), "gain length must equal neurons");

  check_cuda(cudaSetDevice(state.get_device()), "cudaSetDevice");
  auto output = at::empty_like(state);
  auto state_derivative = at::empty_like(state);
  auto current_derivative = at::empty_like(state);
  const int64_t count = state.numel();
  const int blocks = static_cast<int>(
      std::min<int64_t>((count + kThreads - 1) / kThreads, 4096));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  advance_train_forward_kernel<<<blocks, kThreads, 0, stream>>>(
      state.data_ptr<float>(),
      current.data_ptr<float>(),
      alpha.data_ptr<float>(),
      gain.data_ptr<float>(),
      output.data_ptr<float>(),
      state_derivative.data_ptr<float>(),
      current_derivative.data_ptr<float>(),
      count,
      state.size(1),
      current.stride(0),
      current.stride(1),
      static_cast<float>(drive));
  check_cuda(cudaGetLastError(), "training advance kernel launch");
  return {output, state_derivative, current_derivative};
}

std::vector<at::Tensor> advance_train_backward_cuda(
    at::Tensor grad_output,
    at::Tensor state_derivative,
    at::Tensor current_derivative,
    int64_t stream_handle) {
  check_float_cuda(grad_output, "grad_output");
  check_float_cuda(state_derivative, "state_derivative");
  check_float_cuda(current_derivative, "current_derivative");
  TORCH_CHECK(
      grad_output.sizes() == state_derivative.sizes(),
      "state derivative shape must match grad_output");
  TORCH_CHECK(
      grad_output.sizes() == current_derivative.sizes(),
      "current derivative shape must match grad_output");

  check_cuda(cudaSetDevice(grad_output.get_device()), "cudaSetDevice");
  auto grad_state = at::empty_like(grad_output);
  auto grad_current = at::empty_like(grad_output);
  const int64_t count = grad_output.numel();
  const int blocks = static_cast<int>(
      std::min<int64_t>((count + kThreads - 1) / kThreads, 4096));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  advance_train_backward_kernel<<<blocks, kThreads, 0, stream>>>(
      grad_output.data_ptr<float>(),
      state_derivative.data_ptr<float>(),
      current_derivative.data_ptr<float>(),
      grad_state.data_ptr<float>(),
      grad_current.data_ptr<float>(),
      count);
  check_cuda(cudaGetLastError(), "training advance backward kernel launch");
  return {grad_state, grad_current};
}

at::Tensor canonical_spmm_cuda(
    at::Tensor state,
    at::Tensor in_rowptr,
    at::Tensor in_src,
    at::Tensor weight,
    int64_t blocks,
    int64_t stream_handle) {
  check_float_cuda(state, "state");
  check_float_cuda(weight, "weight");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, neurons]");
  TORCH_CHECK(in_rowptr.scalar_type() == at::kInt, "in_rowptr must be int32");
  TORCH_CHECK(in_src.scalar_type() == at::kInt, "in_src must be int32");
  TORCH_CHECK(in_rowptr.numel() == state.size(1) + 1, "invalid in_rowptr length");
  TORCH_CHECK(weight.numel() == in_src.numel(), "weight and in_src lengths differ");
  TORCH_CHECK(blocks > 0, "blocks must be positive");

  check_cuda(cudaSetDevice(state.get_device()), "cudaSetDevice");
  auto output = at::empty_like(state);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  canonical_spmm_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      state.data_ptr<float>(),
      in_rowptr.data_ptr<int32_t>(),
      in_src.data_ptr<int32_t>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      state.size(1),
      state.numel());
  check_cuda(cudaGetLastError(), "canonical SpMM kernel launch");
  return output;
}

at::Tensor canonical_step_cuda(
    at::Tensor state,
    at::Tensor in_rowptr,
    at::Tensor in_src,
    at::Tensor weight,
    at::Tensor alpha,
    at::Tensor gain,
    double drive,
    int64_t blocks,
    int64_t stream_handle) {
  check_float_cuda(state, "state");
  check_float_cuda(weight, "weight");
  check_float_cuda(alpha, "alpha");
  check_float_cuda(gain, "gain");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, neurons]");
  TORCH_CHECK(in_rowptr.scalar_type() == at::kInt, "in_rowptr must be int32");
  TORCH_CHECK(in_src.scalar_type() == at::kInt, "in_src must be int32");
  TORCH_CHECK(in_rowptr.numel() == state.size(1) + 1, "invalid in_rowptr length");
  TORCH_CHECK(weight.numel() == in_src.numel(), "weight and in_src lengths differ");
  TORCH_CHECK(alpha.numel() == state.size(1), "alpha length must equal neurons");
  TORCH_CHECK(gain.numel() == state.size(1), "gain length must equal neurons");
  TORCH_CHECK(blocks > 0, "blocks must be positive");

  check_cuda(cudaSetDevice(state.get_device()), "cudaSetDevice");
  auto output = at::empty_like(state);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  canonical_step_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      state.data_ptr<float>(),
      in_rowptr.data_ptr<int32_t>(),
      in_src.data_ptr<int32_t>(),
      weight.data_ptr<float>(),
      alpha.data_ptr<float>(),
      gain.data_ptr<float>(),
      output.data_ptr<float>(),
      state.size(1),
      state.numel(),
      static_cast<float>(drive));
  check_cuda(cudaGetLastError(), "canonical fused-step kernel launch");
  return output;
}

at::Tensor canonical_modulated_step_cuda(
    at::Tensor state,
    at::Tensor in_rowptr,
    at::Tensor in_src,
    at::Tensor weight,
    at::Tensor alpha,
    at::Tensor gain,
    at::Tensor drive_shift,
    at::Tensor gain_scale,
    double drive,
    int64_t blocks,
    int64_t stream_handle) {
  check_float_cuda(state, "state");
  check_float_cuda(weight, "weight");
  check_float_cuda(alpha, "alpha");
  check_float_cuda(gain, "gain");
  check_float_cuda(drive_shift, "drive_shift");
  check_float_cuda(gain_scale, "gain_scale");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, neurons]");
  TORCH_CHECK(in_rowptr.scalar_type() == at::kInt, "in_rowptr must be int32");
  TORCH_CHECK(in_src.scalar_type() == at::kInt, "in_src must be int32");
  TORCH_CHECK(in_rowptr.numel() == state.size(1) + 1, "invalid in_rowptr length");
  TORCH_CHECK(weight.numel() == in_src.numel(), "weight and in_src lengths differ");
  TORCH_CHECK(alpha.numel() == state.size(1), "alpha length must equal neurons");
  TORCH_CHECK(gain.numel() == state.size(1), "gain length must equal neurons");
  TORCH_CHECK(drive_shift.sizes() == state.sizes(), "drive_shift shape mismatch");
  TORCH_CHECK(gain_scale.sizes() == state.sizes(), "gain_scale shape mismatch");
  TORCH_CHECK(blocks > 0, "blocks must be positive");

  check_cuda(cudaSetDevice(state.get_device()), "cudaSetDevice");
  auto output = at::empty_like(state);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  canonical_modulated_step_kernel<<<
      static_cast<int>(blocks), kThreads, 0, stream>>>(
      state.data_ptr<float>(),
      in_rowptr.data_ptr<int32_t>(),
      in_src.data_ptr<int32_t>(),
      weight.data_ptr<float>(),
      alpha.data_ptr<float>(),
      gain.data_ptr<float>(),
      drive_shift.data_ptr<float>(),
      gain_scale.data_ptr<float>(),
      output.data_ptr<float>(),
      state.size(1),
      state.numel(),
      static_cast<float>(drive));
  check_cuda(cudaGetLastError(), "canonical modulated-step kernel launch");
  return output;
}

void pels_event_step_cuda(
    at::Tensor state,
    at::Tensor current,
    at::Tensor emitted,
    at::Tensor alpha,
    at::Tensor gain,
    at::Tensor out_rowptr,
    at::Tensor dst,
    at::Tensor out_weight_index,
    at::Tensor weight,
    at::Tensor theta,
    at::Tensor active_index,
    at::Tensor active_delta,
    at::Tensor active_count,
    double drive,
    int64_t scatter_blocks,
    int64_t stream_handle) {
  check_float_cuda(state, "state");
  check_float_cuda(current, "current");
  check_float_cuda(emitted, "emitted");
  check_float_cuda(alpha, "alpha");
  check_float_cuda(gain, "gain");
  check_float_cuda(weight, "weight");
  check_float_cuda(theta, "theta");
  check_float_cuda(active_delta, "active_delta");
  TORCH_CHECK(out_rowptr.scalar_type() == at::kInt, "out_rowptr must be int32");
  TORCH_CHECK(dst.scalar_type() == at::kInt, "dst must be int32");
  TORCH_CHECK(out_weight_index.scalar_type() == at::kInt, "out_weight_index must be int32");
  TORCH_CHECK(active_index.scalar_type() == at::kLong, "active_index must be int64");
  TORCH_CHECK(active_count.scalar_type() == at::kInt, "active_count must be int32");
  TORCH_CHECK(theta.numel() == state.size(1), "theta length must equal neurons");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, neurons]");
  TORCH_CHECK(current.sizes() == state.sizes(), "current shape must match state");
  TORCH_CHECK(emitted.sizes() == state.sizes(), "emitted shape must match state");
  TORCH_CHECK(active_index.numel() >= state.numel(), "active_index workspace is too small");
  TORCH_CHECK(active_delta.numel() >= state.numel(), "active_delta workspace is too small");
  TORCH_CHECK(active_count.numel() == 1, "active_count must contain one int32");

  check_cuda(cudaSetDevice(state.get_device()), "cudaSetDevice");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  check_cuda(
      cudaMemsetAsync(active_count.data_ptr<int>(), 0, sizeof(int), stream),
      "active-count reset");
  const int64_t count = state.numel();
  const int advance_blocks = static_cast<int>(
      std::min<int64_t>((count + kThreads - 1) / kThreads, 4096));
  advance_compact_kernel<<<advance_blocks, kThreads, 0, stream>>>(
      state.data_ptr<float>(),
      current.data_ptr<float>(),
      emitted.data_ptr<float>(),
      alpha.data_ptr<float>(),
      gain.data_ptr<float>(),
      theta.data_ptr<float>(),
      active_index.data_ptr<int64_t>(),
      active_delta.data_ptr<float>(),
      active_count.data_ptr<int>(),
      count,
      state.size(1),
      static_cast<float>(drive));
  check_cuda(cudaGetLastError(), "advance/compact kernel launch");

  const int blocks = static_cast<int>(std::max<int64_t>(1, scatter_blocks));
  scatter_active_edges_kernel<<<blocks, kThreads, 0, stream>>>(
      current.data_ptr<float>(),
      out_rowptr.data_ptr<int32_t>(),
      dst.data_ptr<int32_t>(),
      out_weight_index.data_ptr<int32_t>(),
      weight.data_ptr<float>(),
      active_index.data_ptr<int64_t>(),
      active_delta.data_ptr<float>(),
      active_count.data_ptr<int>(),
      state.size(1));
  check_cuda(cudaGetLastError(), "event-scatter kernel launch");
}

at::Tensor pels_dopamine_event_step_cuda(
    at::Tensor state,
    at::Tensor current,
    at::Tensor emitted,
    at::Tensor alpha,
    at::Tensor gain,
    at::Tensor drive_shift,
    at::Tensor gain_scale,
    at::Tensor out_rowptr,
    at::Tensor dst,
    at::Tensor out_weight_index,
    at::Tensor weight,
    at::Tensor theta,
    at::Tensor active_index,
    at::Tensor active_delta,
    at::Tensor active_count,
    double drive,
    int64_t scatter_blocks,
    int64_t stream_handle) {
  check_float_cuda(state, "state");
  check_float_cuda(current, "current");
  check_float_cuda(emitted, "emitted");
  check_float_cuda(alpha, "alpha");
  check_float_cuda(gain, "gain");
  check_float_cuda(drive_shift, "drive_shift");
  check_float_cuda(gain_scale, "gain_scale");
  check_float_cuda(weight, "weight");
  check_float_cuda(theta, "theta");
  check_float_cuda(active_delta, "active_delta");
  TORCH_CHECK(out_rowptr.scalar_type() == at::kInt, "out_rowptr must be int32");
  TORCH_CHECK(dst.scalar_type() == at::kInt, "dst must be int32");
  TORCH_CHECK(out_weight_index.scalar_type() == at::kInt, "out_weight_index must be int32");
  TORCH_CHECK(active_index.scalar_type() == at::kLong, "active_index must be int64");
  TORCH_CHECK(active_count.scalar_type() == at::kInt, "active_count must be int32");
  TORCH_CHECK(state.dim() == 2, "state must have shape [batch, neurons]");
  TORCH_CHECK(current.sizes() == state.sizes(), "current shape must match state");
  TORCH_CHECK(emitted.sizes() == state.sizes(), "emitted shape must match state");
  TORCH_CHECK(drive_shift.sizes() == state.sizes(), "drive_shift shape mismatch");
  TORCH_CHECK(gain_scale.sizes() == state.sizes(), "gain_scale shape mismatch");
  TORCH_CHECK(theta.numel() == state.size(1), "theta length must equal neurons");
  TORCH_CHECK(active_index.numel() >= state.numel(), "active_index workspace is too small");
  TORCH_CHECK(active_delta.numel() >= state.numel(), "active_delta workspace is too small");
  TORCH_CHECK(active_count.numel() == 1, "active_count must contain one int32");

  check_cuda(cudaSetDevice(state.get_device()), "cudaSetDevice");
  auto output = at::empty_like(state);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  check_cuda(
      cudaMemsetAsync(active_count.data_ptr<int>(), 0, sizeof(int), stream),
      "active-count reset");
  const int64_t count = state.numel();
  const int advance_blocks = static_cast<int>(
      std::min<int64_t>((count + kThreads - 1) / kThreads, 4096));
  advance_compact_modulated_kernel<<<advance_blocks, kThreads, 0, stream>>>(
      state.data_ptr<float>(),
      current.data_ptr<float>(),
      emitted.data_ptr<float>(),
      alpha.data_ptr<float>(),
      gain.data_ptr<float>(),
      drive_shift.data_ptr<float>(),
      gain_scale.data_ptr<float>(),
      theta.data_ptr<float>(),
      output.data_ptr<float>(),
      active_index.data_ptr<int64_t>(),
      active_delta.data_ptr<float>(),
      active_count.data_ptr<int>(),
      count,
      state.size(1),
      static_cast<float>(drive));
  check_cuda(cudaGetLastError(), "dopamine advance/compact kernel launch");

  const int blocks = static_cast<int>(std::max<int64_t>(1, scatter_blocks));
  scatter_active_edges_kernel<<<blocks, kThreads, 0, stream>>>(
      current.data_ptr<float>(),
      out_rowptr.data_ptr<int32_t>(),
      dst.data_ptr<int32_t>(),
      out_weight_index.data_ptr<int32_t>(),
      weight.data_ptr<float>(),
      active_index.data_ptr<int64_t>(),
      active_delta.data_ptr<float>(),
      active_count.data_ptr<int>(),
      state.size(1));
  check_cuda(cudaGetLastError(), "dopamine event-scatter kernel launch");
  return output;
}

void update_eligibility_cuda(
    at::Tensor pre_trace,
    at::Tensor post_trace,
    at::Tensor eligibility,
    at::Tensor previous_state,
    at::Tensor next_state,
    at::Tensor src,
    at::Tensor dst,
    double pre_decay,
    double post_decay,
    double eligibility_decay,
    double potentiation_scale,
    double depression_scale,
    int64_t stream_handle) {
  check_float_cuda(pre_trace, "pre_trace");
  check_float_cuda(post_trace, "post_trace");
  check_float_cuda(eligibility, "eligibility");
  check_float_cuda(previous_state, "previous_state");
  check_float_cuda(next_state, "next_state");
  TORCH_CHECK(src.scalar_type() == at::kInt, "src must be int32");
  TORCH_CHECK(dst.scalar_type() == at::kInt, "dst must be int32");
  TORCH_CHECK(previous_state.dim() == 2, "state must have shape [batch, neurons]");
  TORCH_CHECK(pre_trace.sizes() == previous_state.sizes(), "pre_trace shape mismatch");
  TORCH_CHECK(post_trace.sizes() == previous_state.sizes(), "post_trace shape mismatch");
  TORCH_CHECK(next_state.sizes() == previous_state.sizes(), "next_state shape mismatch");
  TORCH_CHECK(src.numel() == eligibility.numel(), "src and eligibility lengths differ");
  TORCH_CHECK(dst.numel() == eligibility.numel(), "dst and eligibility lengths differ");

  check_cuda(cudaSetDevice(previous_state.get_device()), "cudaSetDevice");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  const int64_t state_count = previous_state.numel();
  const int trace_blocks = static_cast<int>(
      std::min<int64_t>((state_count + kThreads - 1) / kThreads, 4096));
  update_neuron_traces_kernel<<<trace_blocks, kThreads, 0, stream>>>(
      pre_trace.data_ptr<float>(),
      post_trace.data_ptr<float>(),
      previous_state.data_ptr<float>(),
      next_state.data_ptr<float>(),
      state_count,
      static_cast<float>(pre_decay),
      static_cast<float>(post_decay));
  check_cuda(cudaGetLastError(), "neuron trace kernel launch");

  const int64_t edges = eligibility.numel();
  const int edge_blocks = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>((edges + kThreads - 1) / kThreads, 4096)));
  update_edge_eligibility_kernel<<<edge_blocks, kThreads, 0, stream>>>(
      pre_trace.data_ptr<float>(),
      post_trace.data_ptr<float>(),
      eligibility.data_ptr<float>(),
      previous_state.data_ptr<float>(),
      next_state.data_ptr<float>(),
      src.data_ptr<int32_t>(),
      dst.data_ptr<int32_t>(),
      edges,
      previous_state.size(1),
      previous_state.size(0),
      static_cast<float>(eligibility_decay),
      static_cast<float>(potentiation_scale),
      static_cast<float>(depression_scale));
  check_cuda(cudaGetLastError(), "edge eligibility kernel launch");
}

at::Tensor dopamine_weight_update_cuda(
    at::Tensor source_weight,
    at::Tensor destination_weight,
    at::Tensor out_weight_index,
    at::Tensor eligibility,
    at::Tensor phasic_signal,
    at::Tensor dst,
    double learning_rate,
    double weight_decay,
    double minimum_weight,
    double maximum_weight,
    bool preserve_sign,
    bool update_magnitude,
    int64_t stream_handle) {
  check_float_cuda(source_weight, "source_weight");
  check_float_cuda(destination_weight, "destination_weight");
  check_float_cuda(eligibility, "eligibility");
  check_float_cuda(phasic_signal, "phasic_signal");
  TORCH_CHECK(out_weight_index.scalar_type() == at::kInt, "out_weight_index must be int32");
  TORCH_CHECK(dst.scalar_type() == at::kInt, "dst must be int32");
  TORCH_CHECK(source_weight.dim() == 1, "source_weight must be rank-1");
  TORCH_CHECK(source_weight.numel() == destination_weight.numel(), "weight view lengths differ");
  TORCH_CHECK(source_weight.numel() == out_weight_index.numel(), "weight index length differs");
  TORCH_CHECK(source_weight.numel() == eligibility.numel(), "eligibility length differs");
  TORCH_CHECK(source_weight.numel() == dst.numel(), "dst length differs");
  TORCH_CHECK(phasic_signal.dim() == 2 && phasic_signal.size(0) == 1,
      "exact native online plasticity requires phasic_signal shape [1, N]");
  TORCH_CHECK(learning_rate >= 0.0, "learning_rate must be non-negative");
  TORCH_CHECK(weight_decay >= 0.0 && weight_decay <= 1.0,
      "weight_decay must lie in [0, 1]");
  TORCH_CHECK(minimum_weight <= maximum_weight, "weight bounds are inconsistent");

  check_cuda(cudaSetDevice(source_weight.get_device()), "cudaSetDevice");
  auto maximum_delta = at::zeros({1}, source_weight.options());
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  const int64_t edges = source_weight.numel();
  const int blocks = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>((edges + kThreads - 1) / kThreads, 4096)));
  dopamine_weight_update_kernel<<<blocks, kThreads, 0, stream>>>(
      source_weight.data_ptr<float>(),
      destination_weight.data_ptr<float>(),
      out_weight_index.data_ptr<int32_t>(),
      eligibility.data_ptr<float>(),
      phasic_signal.data_ptr<float>(),
      dst.data_ptr<int32_t>(),
      edges,
      static_cast<float>(learning_rate),
      static_cast<float>(weight_decay),
      static_cast<float>(minimum_weight),
      static_cast<float>(maximum_weight),
      preserve_sign,
      update_magnitude,
      reinterpret_cast<int*>(maximum_delta.data_ptr<float>()));
  check_cuda(cudaGetLastError(), "dopamine weight-update kernel launch");
  return maximum_delta;
}
