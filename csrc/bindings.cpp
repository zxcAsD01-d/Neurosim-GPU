#include <torch/extension.h>

std::vector<torch::Tensor> dendritic_step_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor>,
    torch::Tensor, torch::Tensor, torch::Tensor,
    std::optional<torch::Tensor>, double, std::optional<torch::Tensor>, double, int64_t,
    std::optional<torch::Tensor>, std::optional<torch::Tensor>);
std::vector<torch::Tensor> dendritic_event_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor>,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    std::optional<torch::Tensor>, double, std::optional<torch::Tensor>, double);

torch::Tensor advance_cuda(
    torch::Tensor state,
    torch::Tensor current,
    torch::Tensor alpha,
    torch::Tensor gain,
    double drive,
    int64_t stream_handle);

std::vector<torch::Tensor> advance_train_forward_cuda(
    torch::Tensor state,
    torch::Tensor current,
    torch::Tensor alpha,
    torch::Tensor gain,
    double drive,
    int64_t stream_handle);

std::vector<torch::Tensor> advance_train_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor state_derivative,
    torch::Tensor current_derivative,
    int64_t stream_handle);

torch::Tensor canonical_spmm_cuda(
    torch::Tensor state,
    torch::Tensor in_rowptr,
    torch::Tensor in_src,
    torch::Tensor weight,
    int64_t blocks,
    int64_t stream_handle);

torch::Tensor canonical_step_cuda(
    torch::Tensor state,
    torch::Tensor in_rowptr,
    torch::Tensor in_src,
    torch::Tensor weight,
    torch::Tensor alpha,
    torch::Tensor gain,
    double drive,
    int64_t blocks,
    int64_t stream_handle);

torch::Tensor canonical_modulated_step_cuda(
    torch::Tensor state,
    torch::Tensor in_rowptr,
    torch::Tensor in_src,
    torch::Tensor weight,
    torch::Tensor alpha,
    torch::Tensor gain,
    torch::Tensor drive_shift,
    torch::Tensor gain_scale,
    double drive,
    int64_t blocks,
    int64_t stream_handle);

void pels_event_step_cuda(
    torch::Tensor state,
    torch::Tensor current,
    torch::Tensor emitted,
    torch::Tensor alpha,
    torch::Tensor gain,
    torch::Tensor out_rowptr,
    torch::Tensor dst,
    torch::Tensor out_weight_index,
    torch::Tensor weight,
    torch::Tensor theta,
    torch::Tensor active_index,
    torch::Tensor active_delta,
    torch::Tensor active_count,
    double drive,
    int64_t scatter_blocks,
    int64_t stream_handle);

torch::Tensor pels_dopamine_event_step_cuda(
    torch::Tensor state,
    torch::Tensor current,
    torch::Tensor emitted,
    torch::Tensor alpha,
    torch::Tensor gain,
    torch::Tensor drive_shift,
    torch::Tensor gain_scale,
    torch::Tensor out_rowptr,
    torch::Tensor dst,
    torch::Tensor out_weight_index,
    torch::Tensor weight,
    torch::Tensor theta,
    torch::Tensor active_index,
    torch::Tensor active_delta,
    torch::Tensor active_count,
    double drive,
    int64_t scatter_blocks,
    int64_t stream_handle);

void update_eligibility_cuda(
    torch::Tensor pre_trace,
    torch::Tensor post_trace,
    torch::Tensor eligibility,
    torch::Tensor previous_state,
    torch::Tensor next_state,
    torch::Tensor src,
    torch::Tensor dst,
    double pre_decay,
    double post_decay,
    double eligibility_decay,
    double potentiation_scale,
    double depression_scale,
    int64_t stream_handle);

torch::Tensor dopamine_weight_update_cuda(
    torch::Tensor source_weight,
    torch::Tensor destination_weight,
    torch::Tensor out_weight_index,
    torch::Tensor eligibility,
    torch::Tensor phasic_signal,
    torch::Tensor dst,
    double learning_rate,
    double weight_decay,
    double minimum_weight,
    double maximum_weight,
    bool preserve_sign,
    bool update_magnitude,
    int64_t stream_handle);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("dendritic_step", &dendritic_step_cuda, "Fused sparse compartment tick (CUDA)",
    pybind11::arg("state"), pybind11::arg("dendrite"), pybind11::arg("adaptation"),
    pybind11::arg("params"), pybind11::arg("rowptr"), pybind11::arg("source"),
    pybind11::arg("weight"), pybind11::arg("input"), pybind11::arg("input_scalar"),
    pybind11::arg("dendritic_input"), pybind11::arg("dendritic_scalar"), pybind11::arg("width"),
    pybind11::arg("block_scales") = pybind11::none(), pybind11::arg("global_scale") = pybind11::none());
  module.def("dendritic_event", &dendritic_event_cuda, "Compartment update and event scatter (CUDA)");
  module.def("advance", &advance_cuda, "Fused neural state advance (CUDA)");
  module.def(
      "advance_train_forward",
      &advance_train_forward_cuda,
      "Fused neural advance with local derivatives (CUDA)");
  module.def(
      "advance_train_backward",
      &advance_train_backward_cuda,
      "Fused neural local backward (CUDA)");
  module.def(
      "canonical_spmm",
      &canonical_spmm_cuda,
      "Single-weight canonical sparse matrix multiplication (CUDA)");
  module.def(
      "canonical_step",
      &canonical_step_cuda,
      "Canonical SpMM fused with neural dynamics (CUDA)");
  module.def(
      "canonical_modulated_step",
      &canonical_modulated_step_cuda,
      "Canonical SpMM fused with receptor-modulated dynamics (CUDA)");
  module.def(
      "pels_event_step_",
      &pels_event_step_cuda,
      "Fused PELS advance, compact and event scatter (CUDA, in-place)");
  module.def(
      "pels_dopamine_event_step_",
      &pels_dopamine_event_step_cuda,
      "Fused receptor-modulated PELS event step (CUDA)");
  module.def(
      "update_eligibility_",
      &update_eligibility_cuda,
      "Fused neuron traces and edge eligibility update (CUDA, in-place)");
  module.def(
      "dopamine_weight_update_",
      &dopamine_weight_update_cuda,
      "Fused dopamine-gated canonical weight update (CUDA, in-place)");
}
