"""Command-line entry point for the Lite-DPAS COST2100 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train_baseline import train_model


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--compression-ratio", type=int, choices=(4, 16, 32, 64), default=4)
    parser.add_argument("--model", choices=("litedpas_v2", "litedpas_v4", "litedpas_v5"), default="litedpas_v5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--loss-mode", choices=("nmse", "mse"), default="nmse")
    parser.add_argument("--scheduler-patience", type=int, default=25)
    parser.add_argument("--teacher-repo", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--distill-weight", type=float, default=0.25)
    args = parser.parse_args()

    data_dir = args.data_dir or args.project_root / "Data"
    encoded_dim = 2048 // args.compression_ratio
    checkpoint = train_model(
        args.model,
        args.project_root,
        data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        encoded_dim=encoded_dim,
        run_tag=args.run_tag,
        save_every=args.save_every,
        resume_from=args.resume_from,
        loss_mode=args.loss_mode,
        scheduler_patience=args.scheduler_patience,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_repo=args.teacher_repo,
        distill_weight=args.distill_weight,
    )
    print(f"Best validation checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
