from __future__ import annotations

from datetime import timedelta
from time import perf_counter


class Stopwatch:
    def __init__(self) -> None:
        self.start = perf_counter()

    def reset(self) -> None:
        self.start = perf_counter()

    def elapsed(self) -> float:
        return perf_counter() - self.start

    def elapsed_hhmmss(self) -> str:
        total_seconds = int(self.elapsed())
        delta = timedelta(seconds=total_seconds)
        hours = delta.seconds // 3600 + delta.days * 24
        minutes = (delta.seconds % 3600) // 60
        seconds = delta.seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
