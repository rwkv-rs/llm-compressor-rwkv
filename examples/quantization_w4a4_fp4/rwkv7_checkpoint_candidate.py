"""Run one formal RWKV-7 1.5B quantization candidate on Blackwell.

The input calibration JSONL contains one ``{"input_ids": [...]}`` object per
line. Its SHA-256 and the implementation Git OID are mandatory provenance, so a
result cannot silently switch calibration data or source code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmcompressor.modifiers.quantization.rwkv7 import (
    run_rwkv7_checkpoint_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=("nvfp4-w4a4", "nvfp4-w4a16"),
        required=True,
    )
    parser.add_argument("--max-calibration-samples", type=int, default=128)
    parser.add_argument("--max-calibration-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_rwkv7_checkpoint_candidate(
        args.checkpoint,
        args.calibration_jsonl,
        args.output_dir,
        calibration_sha256=args.calibration_sha256,
        implementation_revision=args.implementation_revision,
        candidate=args.candidate,
        max_calibration_samples=args.max_calibration_samples,
        max_calibration_length=args.max_calibration_length,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
