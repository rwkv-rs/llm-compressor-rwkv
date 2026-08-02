"""Build a formal RWKV-7 Pareto artifact from fully bound evaluation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmcompressor.modifiers.quantization.rwkv7_evaluation import (
    build_rwkv7_pareto_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_rwkv7_pareto_artifact(
        args.contract,
        args.evidence,
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
