from __future__ import annotations

import argparse
import json

from neurosim_gpu.cost_model import CostModel


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only theoretical cost model")
    parser.add_argument("--neurons", type=int, default=166_000)
    parser.add_argument("--edges", type=int, default=125_000_000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--regions", type=int, default=256)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--region-edges", type=int, default=1024)
    parser.add_argument("--chemistry-interval", type=int, default=10)
    parser.add_argument("--plasticity-interval", type=int, default=10)
    parser.add_argument("--dopamine-delay", type=int, default=4)
    parser.add_argument("--axon-delay", type=int, default=8)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--activity", type=float, default=0.05)
    parser.add_argument("--candidates", type=int, default=2_000_000)
    parser.add_argument("--migration-interval", type=int, default=10_000)
    parser.add_argument("--structural-interval", type=int, default=100_000)
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--eligibility", action="store_true")
    parser.add_argument("--plasticity", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = CostModel(
        n_neurons=args.neurons, n_edges=args.edges, batch=args.batch,
        n_regions=args.regions, n_sources=args.sources,
        region_edges=args.region_edges,
        chemistry_interval=args.chemistry_interval,
        plasticity_interval=args.plasticity_interval,
        max_dopamine_delay=args.dopamine_delay,
        max_axon_delay=args.axon_delay,
        communication_channels=args.channels,
        active_edge_fraction=args.activity,
        candidate_edges=args.candidates,
        migration_interval=args.migration_interval,
        structural_interval=args.structural_interval,
        include_communication=args.communication,
        include_eligibility=args.eligibility,
        include_plasticity=args.plasticity,
        include_development=args.development,
    ).as_dict()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key in (
            "base_work", "event_work", "eligibility_work", "full_matrix_work",
            "full_event_work", "event_to_matrix_work_ratio", "slim_inference_bytes",
            "mutable_online_bytes", "estimated_persistent_bytes",
        ):
            print(f"{key}={report[key]}")


if __name__ == "__main__":
    main()
