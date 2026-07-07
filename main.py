from __future__ import annotations

import sys
from pathlib import Path

from train import main as train_main


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    if len(argv) > 1 and argv[1] == "predict":
        from predict import main as predict_main
        return predict_main()

    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())
