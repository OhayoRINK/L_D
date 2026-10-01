"""Train Lite-DPAS V5 on the COST2100 outdoor scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train_baseline import train_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--compression-ratio", type=int, choices=(4, 16, 32, 64), default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", type=str, default="outdoor_v5_01")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=25)
    args = parser.parse_args()

    data_dir = args.data_dir or args.project_root / "Data"
    encoded_dim = 2048 // args.compression_ratio
    checkpoint = train_model(
        "litedpas_v5",
        args.project_root,
        data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        encoded_dim=encoded_dim,
        run_tag=args.run_tag,
        save_every=args.save_every,
        resume_from=args.resume_from,
        scenario="outdoor",
        loss_mode="nmse",
        scheduler_patience=args.scheduler_patience,
    )
    print(f"Best outdoor validation checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
