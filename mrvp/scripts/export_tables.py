from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = Path(args.results)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for tex in src.glob("**/*.tex"):
        target = out / tex.name
        shutil.copy2(tex, target)


if __name__ == "__main__":
    main()
