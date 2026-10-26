from __future__ import annotations

import torch

from neurosim_gpu import (
    DopamineMatrixRuntime,
    DopaminePlasticityConfig,
    DopamineReceptors,
    Dynamics,
    HybridPELSRuntime,
    MatrixRuntime,
    NativeDopamineMatrixRuntime,
    NativeDopamineEventRuntime,
    DopaminePersistentRuntime,
    RegionalDopamineField,
    random_topology,
)
from neurosim_gpu.native import (
    NativeCanonicalLinear,
    NativeCanonicalMatrixRuntime,
    NativeEventPELSRuntime,
    NativeMatrixRuntime,
    load_native_extension,
    native_advance_training,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    extension = load_native_extension(verbose=False)
    device = torch.device("cuda")
    n = 4096
    topology = random_topology(n, 100_000, device=device, seed=23)
    dynamics = Dynamics(
        alpha=torch.full((n,), 0.95, device=device),
        gain=torch.full((n,), 0.8, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(29)
    initial = torch.rand(2, n, device=device, generator=generator)

    matrix = MatrixRuntime(topology, dynamics, initial.clone())
    native_matrix = NativeMatrixRuntime(
        topology, dynamics, initial.clone(), extension=extension
    )
    canonical_linear = NativeCanonicalLinear(
        topology, initial.clone(), extension=extension
    )
    canonical_matrix = NativeCanonicalMatrixRuntime(
        topology, dynamics, initial.clone(), extension=extension
    )
    pels = HybridPELSRuntime(
        topology,
        dynamics,
        initial.clone(),
        theta=0.02,
        propagation="event",
    )
    native_pels = NativeEventPELSRuntime(
        topology,
        dynamics,
        initial.clone(),
        theta=0.02,
        extension=extension,
    )

    with torch.inference_mode():
        expected_linear = matrix._propagate(initial)
        actual_linear = canonical_linear.step()
        for _ in range(30):
            expected_matrix = matrix.step(0.15)
            actual_matrix = native_matrix.step(0.15)
            actual_canonical = canonical_matrix.step(0.15)
            expected_pels = pels.step(0.15)
            actual_pels = native_pels.step(0.15)
        torch.cuda.synchronize()

    torch.testing.assert_close(actual_linear, expected_linear, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_matrix, expected_matrix, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        actual_canonical, expected_matrix, rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(actual_pels, expected_pels, rtol=1e-3, atol=1e-4)
    recomputed_current = native_pels._propagate(
        native_pels.emitted_state
    ).contiguous()
    torch.testing.assert_close(
        native_pels.persistent_current,
        recomputed_current,
        rtol=2e-4,
        atol=2e-5,
    )

    reference_state = initial.clone().requires_grad_(True)
    reference_current = torch.randn_like(initial, requires_grad=True)
    reference_output = dynamics.advance(
        reference_state, reference_current, 0.15
    )
    reference_gradients = torch.autograd.grad(
        reference_output.square().mean(),
        (reference_state, reference_current),
    )
    native_state = initial.clone().requires_grad_(True)
    native_current = reference_current.detach().clone().requires_grad_(True)
    native_output = native_advance_training(
        native_state,
        native_current,
        dynamics,
        0.15,
        extension,
    )
    native_gradients = torch.autograd.grad(
        native_output.square().mean(),
        (native_state, native_current),
    )
    torch.testing.assert_close(
        native_output, reference_output, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        native_gradients[0], reference_gradients[0], rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(
        native_gradients[1], reference_gradients[1], rtol=3e-5, atol=3e-6
    )

    regions = 8
    region = torch.arange(regions, device=device)
    projection = torch.zeros(2, regions, device=device)
    projection[region.remainder(2), region] = 1.0
    field = RegionalDopamineField(
        tonic_baseline=torch.full((regions,), 0.5, device=device),
        uptake_rate=torch.full((regions,), 0.1, device=device),
        decay_rate=torch.full((regions,), 0.02, device=device),
        release_projection=projection,
        edge_src=region.to(torch.int64),
        edge_dst=torch.roll(region, -1).to(torch.int64),
        diffusion_rate=torch.full((regions,), 0.04, device=device),
        dt=0.1,
    )
    d1 = (torch.arange(n, device=device) % 2 == 0).float()
    receptors = DopamineReceptors(
        d1_density=d1,
        d2_density=1.0 - d1,
        tonic_concentration=0.5,
        d1_half_saturation=0.5,
        d2_half_saturation=0.25,
    )
    config = DopaminePlasticityConfig(
        track_eligibility=True,
        chemistry_interval=2,
    )
    neuron_region = torch.arange(n, device=device).remainder(regions)
    reference_da = DopamineMatrixRuntime(
        topology,
        dynamics,
        initial.clone(),
        chemistry=field,
        receptors=receptors,
        neuron_region=neuron_region,
        config=config,
    )
    native_da = NativeDopamineMatrixRuntime(
        topology,
        dynamics,
        initial.clone(),
        chemistry=field,
        receptors=receptors,
        neuron_region=neuron_region,
        config=config,
        extension=extension,
    )
    with torch.inference_mode():
        for value in (0.5, 0.0, -0.25, 0.0, 0.1, 0.0):
            rpe = torch.full((2, 2), value, device=device)
            reference_da_state = reference_da.step(0.15, rpe=rpe)
            native_da_state = native_da.step(0.15, rpe=rpe)
    torch.testing.assert_close(
        native_da_state, reference_da_state, rtol=3e-4, atol=3e-5
    )
    torch.testing.assert_close(
        native_da.concentration,
        reference_da.concentration,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        native_da.eligibility,
        reference_da.eligibility,
        rtol=3e-4,
        atol=3e-5,
    )
    reference_da_event = DopaminePersistentRuntime(
        topology,
        dynamics,
        initial.clone(),
        chemistry=field,
        receptors=receptors,
        neuron_region=neuron_region,
        theta=0.02,
        config=config,
    )
    native_da_event = NativeDopamineEventRuntime(
        topology,
        dynamics,
        initial.clone(),
        chemistry=field,
        receptors=receptors,
        neuron_region=neuron_region,
        theta=0.02,
        config=config,
        extension=extension,
    )
    with torch.inference_mode():
        for value in (0.5, 0.0, -0.25, 0.0, 0.1, 0.0):
            rpe = torch.full((2, 2), value, device=device)
            reference_event_state = reference_da_event.step(0.15, rpe=rpe)
            native_event_state = native_da_event.step(0.15, rpe=rpe)
    torch.testing.assert_close(
        native_event_state, reference_event_state, rtol=2e-3, atol=2e-4
    )
    native_event_current = native_da_event._propagate(
        native_da_event.emitted_state
    ).contiguous()
    torch.testing.assert_close(
        native_da_event.persistent_current,
        native_event_current,
        rtol=4e-4,
        atol=4e-5,
    )

    source_weight = topology.weight.detach().clone()
    destination_weight = topology.in_weight.detach().clone()
    eligibility = torch.linspace(
        -0.5, 0.5, topology.n_edges, device=device
    )
    phasic = torch.linspace(-0.2, 0.2, n, device=device).unsqueeze(0)
    previous = source_weight.clone()
    gate = phasic[0].index_select(0, topology.dst.to(torch.int64))
    expected_weight = previous * 0.99 + (
        0.01 * torch.sign(previous) * gate * eligibility
    )
    expected_weight = torch.where(
        previous > 0,
        torch.clamp(expected_weight, min=0.0, max=1.0),
        torch.where(
            previous < 0,
            torch.clamp(expected_weight, min=-1.0, max=0.0),
            torch.zeros_like(expected_weight),
        ),
    )
    maximum_delta = extension.dopamine_weight_update_(
        source_weight,
        destination_weight,
        topology.out_weight_index,
        eligibility,
        phasic,
        topology.dst,
        0.01,
        0.01,
        -1.0,
        1.0,
        True,
        True,
        torch.cuda.current_stream(device).cuda_stream,
    )
    torch.testing.assert_close(
        source_weight, expected_weight, rtol=2e-6, atol=2e-7
    )
    torch.testing.assert_close(
        destination_weight[
            topology.out_weight_index.to(torch.int64)
        ],
        source_weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        maximum_delta,
        (source_weight - previous).abs().max().reshape(1),
        rtol=2e-6,
        atol=2e-7,
    )

    learning_n = 512
    learning_base = random_topology(
        learning_n, 8_000, device=device, seed=37
    )
    from neurosim_gpu import SparseTopology

    reference_topology = SparseTopology.from_edges(
        learning_n,
        learning_base.src.to(torch.int64),
        learning_base.dst.to(torch.int64),
        learning_base.weight.detach().clone(),
    )
    native_topology = SparseTopology.from_edges(
        learning_n,
        learning_base.src.to(torch.int64),
        learning_base.dst.to(torch.int64),
        learning_base.weight.detach().clone(),
    )
    learning_dynamics = Dynamics(
        alpha=torch.full((learning_n,), 0.95, device=device),
        gain=torch.full((learning_n,), 0.8, device=device),
    )
    learning_initial = torch.rand(1, learning_n, device=device)
    learning_regions = 4
    learning_region = torch.arange(learning_regions, device=device)
    learning_projection = torch.ones(1, learning_regions, device=device)
    learning_field = RegionalDopamineField(
        tonic_baseline=torch.full((learning_regions,), 0.5, device=device),
        uptake_rate=torch.full((learning_regions,), 0.1, device=device),
        decay_rate=torch.full((learning_regions,), 0.02, device=device),
        release_projection=learning_projection,
        edge_src=learning_region.to(torch.int64),
        edge_dst=torch.roll(learning_region, -1).to(torch.int64),
        diffusion_rate=torch.full((learning_regions,), 0.04, device=device),
        dt=0.1,
    )
    learning_d1 = (
        torch.arange(learning_n, device=device) % 2 == 0
    ).float()
    learning_receptors = DopamineReceptors(
        d1_density=learning_d1,
        d2_density=1.0 - learning_d1,
        tonic_concentration=0.5,
        d1_half_saturation=0.5,
        d2_half_saturation=0.25,
    )
    learning_config = DopaminePlasticityConfig(
        learning_rate=0.01,
        chemistry_interval=1,
        validate_inputs=False,
    )
    learning_map = torch.arange(
        learning_n, device=device
    ).remainder(learning_regions)
    reference_learning = DopamineMatrixRuntime(
        reference_topology,
        learning_dynamics,
        learning_initial.clone(),
        chemistry=learning_field,
        receptors=learning_receptors,
        neuron_region=learning_map,
        config=learning_config,
    )
    native_learning = NativeDopamineMatrixRuntime(
        native_topology,
        learning_dynamics,
        learning_initial.clone(),
        chemistry=learning_field,
        receptors=learning_receptors,
        neuron_region=learning_map,
        config=learning_config,
        extension=extension,
    )
    with torch.inference_mode():
        for value in (0.5, 0.0, -0.25, 0.1, 0.0):
            rpe = torch.full((1, 1), value, device=device)
            reference_learning_state = reference_learning.step(0.15, rpe=rpe)
            native_learning_state = native_learning.step(0.15, rpe=rpe)
    torch.testing.assert_close(
        native_learning_state,
        reference_learning_state,
        rtol=8e-4,
        atol=8e-5,
    )
    torch.testing.assert_close(
        native_learning.eligibility,
        reference_learning.eligibility,
        rtol=8e-4,
        atol=8e-5,
    )
    torch.testing.assert_close(
        native_topology.weight,
        reference_topology.weight,
        rtol=8e-4,
        atol=8e-5,
    )
    torch.testing.assert_close(
        native_topology.in_weight[
            native_topology.out_weight_index.to(torch.int64)
        ],
        native_topology.weight,
        rtol=0.0,
        atol=0.0,
    )
    if native_topology.weight_version != reference_topology.weight_version:
        raise AssertionError("native and reference weight versions diverged")
    print("native CUDA validation: PASS")


if __name__ == "__main__":
    main()
