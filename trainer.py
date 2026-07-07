from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch

from dataset import Dataset
from globals import G
from model import Model


def _draw_checkpoint_progress(current_step: int, total_steps: int) -> None:
    if total_steps <= 0:
        return
    width = 30
    filled = int(round(width * current_step / total_steps))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    percent = current_step / total_steps
    print(f"\rCheckpoint progress: |{bar}| {current_step}/{total_steps} ({percent:.0%})", end="", flush=True)


class Trainer:
    def __init__(self, model: Model, dataset: Dataset, device: torch.device) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=G.learning_rate)

    def train(
        self,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 100,
        max_iter: int = 5000,
    ) -> float:
        self.model.train()
        loss_value = 0.0
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_interval = max(1, int(checkpoint_interval))
        checkpoint_steps = list(range(checkpoint_interval, max_iter + 1, checkpoint_interval))
        if not checkpoint_steps or checkpoint_steps[-1] != max_iter:
            checkpoint_steps.append(max_iter)

        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"{datetime.now().strftime('%y%m%d-%H%M')}.chkpt"
        checkpoint_index = 0
        total_checkpoints = len(checkpoint_steps)

        for iteration in range(max_iter):
            xb, yb = self.dataset.get_batch()
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_value = float(loss.item())

            if checkpoint_path is not None and checkpoint_index < total_checkpoints and (iteration + 1) == checkpoint_steps[checkpoint_index]:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "iteration": iteration + 1,
                        "loss": loss_value,
                    },
                    checkpoint_path,
                )
                checkpoint_index += 1
                _draw_checkpoint_progress(checkpoint_index, total_checkpoints)
                if checkpoint_index >= total_checkpoints:
                    print()
                    break

        return loss_value
