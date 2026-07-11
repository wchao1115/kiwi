from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import Dataset, resolve_dataset_path
from globals import G
from model import Model
from stopwatch import Stopwatch
from trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the language model")
    parser.add_argument("dataset", nargs="?", default=None, help="Path to a text file or directory of .txt files")
    parser.add_argument("--max-iter", type=int, default=5000, help="Number of training iterations")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save a checkpoint every N training iterations",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory where checkpoints should be stored",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    dataset_path = resolve_dataset_path(args.dataset)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else project_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + ("CUDA" if device.type == "cuda" else "CPU"))

    dataset = Dataset(dataset_path)
    print(f"Total {dataset.size()} tokens in the dataset.")
    print(f"Total {dataset.vocab_size()} unique tokens in the dataset vocabulary.")

    model = Model(dataset.vocab_size(), device)
    trainer = Trainer(model, dataset, device)

    sw = Stopwatch()
    print(f"Training for {args.max_iter} iterations with checkpoints every {args.checkpoint_interval} iterations.")
    loss = trainer.train(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        max_iter=args.max_iter,
    )
    print(f"Final loss: {loss}")
    print(f"Training time: {sw.elapsed_hhmmss()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
