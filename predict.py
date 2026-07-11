from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import Dataset, resolve_dataset_path
from model import Model


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text with a trained model")
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint file")
    parser.add_argument("--tokens", type=int, default=3000, help="Number of tokens to generate")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    dataset_path = resolve_dataset_path()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + ("CUDA" if device.type == "cuda" else "CPU"))

    dataset = Dataset(dataset_path)
    model = Model(dataset.vocab_size(), device)

    if checkpoint_path is None:
        checkpoint_candidates = sorted((project_dir / "checkpoints").glob("*.chkpt"))
        if checkpoint_candidates:
            checkpoint_path = checkpoint_candidates[-1]

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")

    print("Generating text...")
    start = torch.zeros((1, 1), dtype=torch.int64, device=device)
    generated = model.generate(start, args.tokens)
    generated_utf8 = dataset.decode(generated[0].cpu())
    print(f"Generated text: {generated_utf8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
