"""Reads the log-variances Uncertainty Weighting actually learned.

UW is only doing something distinct from Equal Weighting if its two
log-variances end up apart: the effective weight on a task is exp(-s), so
equal log-variances mean equal weights and UW has collapsed onto EW. This
reads them straight out of the trained checkpoints -- no training, no
modification.

Run where the checkpoints live (Drive, under Colab), passing that directory.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

SEEDS = (42, 43, 44)
MODEL_NAME = "ModelD_UW"


def read_learned_weights(checkpoint_dir: str | Path, seeds=SEEDS) -> pd.DataFrame:
    checkpoint_dir = Path(checkpoint_dir)
    rows = []
    for seed in seeds:
        path = checkpoint_dir / f"{MODEL_NAME}_seed{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["loss_strategy_state"]
        log_var_species = float(state["log_var_species"].item())
        log_var_freshness = float(state["log_var_freshness"].item())

        weight_species = float(torch.exp(torch.tensor(-log_var_species)))
        weight_freshness = float(torch.exp(torch.tensor(-log_var_freshness)))

        rows.append(
            {
                "seed": seed,
                "epoch": ckpt.get("epoch"),
                "log_var_species": log_var_species,
                "log_var_freshness": log_var_freshness,
                "weight_species": weight_species,
                "weight_freshness": weight_freshness,
                "weight_ratio_species_over_freshness": weight_species / weight_freshness,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="05_Checkpoints")
    parser.add_argument(
        "--out", default="08_Leakage_Ablation/01_uw_weights/uw_learned_weights.csv"
    )
    args = parser.parse_args()

    df = read_learned_weights(args.checkpoint_dir)
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False))
    print()
    ratios = df["weight_ratio_species_over_freshness"]
    print(f"weight ratio across seeds: mean {ratios.mean():.4f}, range "
          f"{ratios.min():.4f} to {ratios.max():.4f}")
    print("A ratio near 1.0 means UW converged on near-equal weights, i.e. it "
          "behaved like Equal Weighting despite being free not to.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
