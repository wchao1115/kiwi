from __future__ import annotations

from pathlib import Path

import torch

from globals import G


class Dataset:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        content = path.read_bytes().decode("utf-8").replace("\r", "")
        self.vocab = sorted(set(content))
        self.vocab_map = {character: index for index, character in enumerate(self.vocab)}
        self.tokens = self.encode(content)

        split = int(self.tokens.size(0) * G.train_test_split)
        self.train_data = self.tokens[:split]
        self.test_data = self.tokens[split:]

        torch.manual_seed(G.random_seed)

    def get_batch(self, for_training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if for_training else self.test_data
        max_start = data.size(0) - G.block_size
        if max_start <= 0:
            raise ValueError("Dataset is too small for the configured block size")

        indices = torch.randint(max_start, (G.batch_size,), dtype=torch.int64)
        x = torch.stack([data[i : i + G.block_size] for i in indices])
        y = torch.stack([data[i + 1 : i + G.block_size + 1] for i in indices])
        return x, y

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.vocab_map[character] for character in text], dtype=torch.int64)

    def decode(self, encoded: torch.Tensor) -> str:
        return "".join(self.vocab[index] for index in encoded.tolist())

    def vocab_size(self) -> int:
        return len(self.vocab)

    def size(self) -> int:
        return self.tokens.size(0)
