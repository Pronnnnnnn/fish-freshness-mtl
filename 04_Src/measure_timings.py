"""Re-measures inference latency and records which GPU produced the numbers.

Latency is the one reported figure that depends on the machine, so it has to
carry the device name with it -- "a single GPU" invites the obvious question.
This re-times every checkpoint and writes the device string alongside, without
touching any metric that is device-independent.

Cheap on purpose: timing needs only a loaded model and a dummy tensor, not a
pass over the test set, so this is minutes rather than the full evaluation.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from models import MultiTaskModel, SpeciesConditionedMTL
from evaluate import (
    count_params,
    load_flat24_model,
    load_multitask_model,
    load_single_task_model,
    measure_inference_time_ms,
)

SEEDS = (42, 43, 44)

# The six configurations of the main experiment, timed together so their
# numbers come from one session on one device.
EXPERIMENTS = (
    {"name": "ModelA_species", "kind": "single", "num_classes": 8},
    {"name": "ModelB_freshness", "kind": "single", "num_classes": 3},
    {"name": "ModelC_flat24", "kind": "flat24"},
    {"name": "ModelD_EW", "kind": "mtl"},
    {"name": "ModelD_UW", "kind": "mtl"},
    {"name": "ModelD_DWA", "kind": "mtl"},
)

# Side experiments live outside EXPERIMENTS so a default run stays the six
# above, but --only can still reach them.
SIDE_EXPERIMENTS = (
    {"name": "ModelD_Cond", "kind": "mtl", "model_factory": SpeciesConditionedMTL},
)

ALL_EXPERIMENTS = EXPERIMENTS + SIDE_EXPERIMENTS


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device == "cuda" else "CPU"


def _load(exp: dict, checkpoint_path: str, device: str):
    if exp["kind"] == "single":
        return load_single_task_model(checkpoint_path, exp["num_classes"], device)
    if exp["kind"] == "flat24":
        return load_flat24_model(checkpoint_path, device)
    # "model_factory" lets a two-head variant be timed under the same protocol.
    return load_multitask_model(
        checkpoint_path, device, model_factory=exp.get("model_factory", MultiTaskModel)
    )


def measure_all(
    checkpoint_dir: str | Path, device: str, seeds=SEEDS, experiments=None
) -> pd.DataFrame:
    checkpoint_dir = Path(checkpoint_dir)
    hardware = device_name(device)
    rows = []
    for exp in experiments if experiments is not None else EXPERIMENTS:
        for seed in seeds:
            run_name = f"{exp['name']}_seed{seed}"
            path = checkpoint_dir / f"{run_name}.pt"
            if not path.exists():
                raise FileNotFoundError(f"missing checkpoint: {path}")
            model = _load(exp, str(path), device)
            timing = measure_inference_time_ms(model, device)
            rows.append(
                {
                    "model": exp["name"],
                    "seed": seed,
                    "params": count_params(model),
                    "inference_ms": timing["mean_ms"],
                    "inference_ms_std": timing["std_ms"],
                    "timing_iterations": timing["n"],
                    "device": hardware,
                }
            )
            print(f"[{run_name}] {timing['mean_ms']:.3f} +/- {timing['std_ms']:.3f} ms")
    return pd.DataFrame(rows)


def patch_long_results(long_path: str | Path, timings: pd.DataFrame) -> pd.DataFrame:
    """Replaces the efficiency rows in a long-format results file in place.

    Metrics that do not depend on the device are left exactly as they were, so
    the accuracy figures already reported stay byte-identical.
    """
    long_df = pd.read_csv(long_path)
    kept = long_df[~((long_df.task == "efficiency") & (long_df.metric.isin(
        ["inference_ms", "inference_ms_std"])))]

    new_rows = []
    for _, r in timings.iterrows():
        for metric in ["inference_ms", "inference_ms_std"]:
            new_rows.append(
                {"model": r["model"], "seed": r["seed"], "task": "efficiency",
                 "metric": metric, "value": r[metric]}
            )
    return pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--only",
        default=None,
        help="time a single model by name instead of all six",
    )
    parser.add_argument("--out", required=True, help="CSV for the timing table")
    parser.add_argument("--patch-long", default=None,
                        help="optional long-format results file whose timing rows to replace")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device_name(device)}\n")

    experiments = EXPERIMENTS
    if args.only:
        experiments = tuple(e for e in ALL_EXPERIMENTS if e["name"] == args.only)
        if not experiments:
            known = ", ".join(e["name"] for e in ALL_EXPERIMENTS)
            raise SystemExit(f"unknown model: {args.only!r}; known models are {known}")

    timings = measure_all(args.checkpoint_dir, device, experiments=experiments)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    timings.to_csv(args.out, index=False)
    print(f"\nsaved timings to {args.out}")

    print("\nper model (mean over seeds):")
    print(timings.groupby("model")[["inference_ms", "inference_ms_std"]].mean().round(3).to_string())

    if args.patch_long:
        patched = patch_long_results(args.patch_long, timings)
        patched.to_csv(args.patch_long, index=False)
        print(f"\npatched timing rows in {args.patch_long}")


if __name__ == "__main__":
    main()
