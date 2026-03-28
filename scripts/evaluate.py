from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from chest_xray_vae.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation YAML config")
    args = parser.parse_args()
    run_evaluation(args.config)


if __name__ == "__main__":
    main()
