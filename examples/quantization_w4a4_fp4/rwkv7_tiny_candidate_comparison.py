"""Run a closed, tiny RWKV-7 save/reload/generate comparison on Blackwell.

This is a deterministic implementation diagnostic, not a LightEval result or a
claim about the best quantization scheme for a trained RWKV-7 checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch

SEED = 20260801
PROMPT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
QUALITY_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
MAX_NEW_TOKENS = 8
WARMUP_RUNS = 2
TIMED_RUNS = 5
VARIANT_ORDER = (
    "fp16-baseline",
    "bf16-baseline",
    "nvfp4-w4a4",
    "nvfp4-w4a16",
)
RANKED_CANDIDATES = ("nvfp4-w4a4", "nvfp4-w4a16")
RANKING_RULE = (
    "standard save, fresh reload, and generate(use_cache=True) must pass",
    "ascending absolute short-sequence CE delta versus the BF16 baseline",
    "ascending mean absolute logits error versus the BF16 baseline",
    "descending generated-token throughput at median end-to-end latency",
    "ascending physical safetensors bytes",
    "closed candidate order as the final deterministic tie-break",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _git_revision() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"revision": revision, "dirty": dirty}


def _framework_versions() -> dict[str, str]:
    return {
        "compressed_tensors": importlib.metadata.version("compressed-tensors"),
        "llmcompressor": importlib.metadata.version("llmcompressor"),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
    }


def _tiny_config():
    from transformers import Rwkv7Config

    return Rwkv7Config(
        vocab_size=32,
        context_length=32,
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        head_size=8,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        use_cache=True,
    )


def _create_canonical_checkpoint(path: Path) -> None:
    from transformers.models.rwkv7 import Rwkv7ForCausalLM

    torch.manual_seed(SEED)
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    with torch.no_grad():
        for block in model.model.blocks:
            block.att.k_k.fill_(1)
    model.save_pretrained(path)


def _save_baseline(canonical: Path, destination: Path, dtype: torch.dtype) -> None:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(canonical, dtype=dtype).eval()
    model.config.dtype = dtype
    model.save_pretrained(destination)


def _calibration_inputs():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"<unk>": 0}
    vocabulary.update({f"token-{token_id}": token_id for token_id in range(1, 32)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    processor = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<unk>",
    )
    dataset = DataLoader(
        [
            {
                "input_ids": torch.tensor([PROMPT_IDS]),
                "attention_mask": torch.ones((1, len(PROMPT_IDS)), dtype=torch.long),
            }
        ],
        batch_size=None,
    )
    return dataset, processor


def _quantize_candidate(
    canonical: Path, output_root: Path, candidate: str
) -> dict[str, object]:
    from transformers import AutoModelForCausalLM

    from llmcompressor.modifiers.quantization.rwkv7 import quantize_rwkv7_oneshot

    dataset, processor = _calibration_inputs()

    def model_factory():
        return (
            AutoModelForCausalLM.from_pretrained(canonical, dtype=torch.bfloat16)
            .to("cuda")
            .eval()
        )

    model, metadata = quantize_rwkv7_oneshot(
        model_factory,
        output_root,
        calibration_dataset=dataset,
        processor=processor,
        forced_candidate=candidate,
    )
    del model
    torch.cuda.empty_cache()
    if not metadata["fresh_reload"]["passed"]:
        raise RuntimeError(f"{candidate} failed the fresh reload contract")
    return {
        "audit": metadata["audit"],
        "cell_forward": metadata["cell_forward"],
        "fresh_reload": metadata["fresh_reload"]["evidence"],
    }


def _checkpoint_inventory(path: Path) -> dict[str, object]:
    from safetensors import safe_open

    files = [
        file
        for file in path.rglob("*")
        if file.is_file() and ".fresh-reload-tmp" not in file.parts
    ]
    safetensors_files = sorted(path.glob("*.safetensors"))
    dtype_counts: Counter[str] = Counter()
    packed_dtypes: Counter[str] = Counter()
    packed_count = 0
    for shard in safetensors_files:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                dtype = str(handle.get_slice(name).get_dtype())
                dtype_counts[dtype] += 1
                if name.endswith(".weight_packed"):
                    packed_count += 1
                    packed_dtypes[dtype] += 1
    return {
        "checkpoint_bytes": sum(file.stat().st_size for file in files),
        "model_safetensors_bytes": sum(
            file.stat().st_size for file in safetensors_files
        ),
        "safetensors_shards": len(safetensors_files),
        "tensor_dtype_counts": dict(sorted(dtype_counts.items())),
        "packed_weight_tensor_count": packed_count,
        "packed_weight_dtype_counts": dict(sorted(packed_dtypes.items())),
    }


def _load_checkpoint(path: Path):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(path)
    runtime_dtype = config.dtype
    if not isinstance(runtime_dtype, torch.dtype):
        raise RuntimeError(f"checkpoint lacks a torch runtime dtype: {runtime_dtype!r}")
    load_kwargs = {"device_map": "cuda", "dtype": runtime_dtype}
    if config.to_dict().get("quantization_config"):
        from transformers.utils.quantization_config import CompressedTensorsConfig

        load_kwargs["quantization_config"] = CompressedTensorsConfig(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    return model.to(dtype=runtime_dtype).eval(), runtime_dtype


def _generate(model, prompt: torch.Tensor) -> torch.Tensor:
    return model.generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.8,
        top_k=8,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=[],
    )


def _assert_generated_contract(generated: torch.Tensor) -> None:
    if generated.shape != (1, len(PROMPT_IDS) + MAX_NEW_TOKENS):
        raise RuntimeError(f"generate returned an unexpected shape: {generated.shape}")
    if generated[0, : len(PROMPT_IDS)].tolist() != PROMPT_IDS:
        raise RuntimeError("generate did not preserve the fixed prompt prefix")


def _evaluate_checkpoint(
    checkpoint: Path, variant: str, result_path: Path, logits_path: Path
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("RWKV-7 candidate comparison requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    model, runtime_dtype = _load_checkpoint(checkpoint)
    prompt = torch.tensor([PROMPT_IDS], device="cuda")

    expected_generated = None
    for _ in range(WARMUP_RUNS):
        torch.manual_seed(SEED)
        with torch.inference_mode():
            generated = _generate(model, prompt)
        _assert_generated_contract(generated)
        expected_generated = generated.tolist()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident_memory = torch.cuda.memory_allocated()

    latencies = []
    for _ in range(TIMED_RUNS):
        torch.manual_seed(SEED)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with torch.inference_mode():
            generated = _generate(model, prompt)
        torch.cuda.synchronize()
        _assert_generated_contract(generated)
        latencies.append((time.perf_counter_ns() - start) / 1e6)
        if generated.tolist() != expected_generated:
            raise RuntimeError("fixed-seed generate output changed between iterations")
    peak_memory = torch.cuda.max_memory_allocated()

    quality_input = torch.tensor([QUALITY_IDS], device="cuda")
    with torch.inference_mode():
        logits = model(quality_input, use_cache=False).logits.float().cpu()
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"{variant} produced non-finite quality logits")
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = quality_input[:, 1:].cpu().contiguous()
    cross_entropy = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).item()
    torch.save(logits, logits_path)

    quantiles = torch.tensor(latencies, dtype=torch.float64).quantile(
        torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    )
    median_seconds = quantiles[1].item() / 1000
    result = {
        "variant": variant,
        "standard_contract": {
            "passed": True,
            "load_api": "AutoModelForCausalLM.from_pretrained",
            "generate_api": "PreTrainedModel.generate",
            "use_cache": True,
            "generated_ids": expected_generated,
        },
        "runtime_dtype": str(runtime_dtype),
        "hardware": {
            "device_name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
        },
        "memory": {
            "resident_before_timed_generate_bytes": resident_memory,
            "peak_during_timed_generate_bytes": peak_memory,
            "incremental_peak_generate_bytes": peak_memory - resident_memory,
        },
        "performance": {
            "prefill_decode_latency_ms": {
                "p10": quantiles[0].item(),
                "p50": quantiles[1].item(),
                "p90": quantiles[2].item(),
            },
            "generated_tokens_per_second_p50": MAX_NEW_TOKENS / median_seconds,
            "total_tokens_per_second_p50": (len(PROMPT_IDS) + MAX_NEW_TOKENS)
            / median_seconds,
        },
        "quality": {
            "short_sequence_cross_entropy_proxy": cross_entropy,
            "short_sequence_perplexity_proxy": math.exp(cross_entropy),
        },
    }
    _atomic_json(result_path, result)


def _run_evaluator(
    checkpoint: Path,
    variant: str,
    result_path: Path,
    logits_path: Path,
    process_tmp: Path,
) -> None:
    environment = dict(os.environ)
    environment["TMPDIR"] = str(process_tmp)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--evaluate-checkpoint",
            str(checkpoint),
            "--variant",
            variant,
            "--result",
            str(result_path),
            "--logits",
            str(logits_path),
        ],
        check=True,
        env=environment,
    )


def _logits_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = actual.float() - reference.float()
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
    }


def _rank_candidates(variants: list[dict]) -> dict[str, object]:
    by_name = {variant["variant"]: variant for variant in variants}
    baseline_ce = by_name["bf16-baseline"]["quality"][
        "short_sequence_cross_entropy_proxy"
    ]

    def ranking_key(candidate: str) -> tuple:
        variant = by_name[candidate]
        quality = variant["quality"]
        performance = variant["performance"]
        storage = variant["storage"]
        return (
            abs(quality["short_sequence_cross_entropy_proxy"] - baseline_ce),
            quality["logits_error_vs_bf16"]["mean_abs"],
            -performance["generated_tokens_per_second_p50"],
            storage["model_safetensors_bytes"],
            RANKED_CANDIDATES.index(candidate),
        )

    ordered = sorted(RANKED_CANDIDATES, key=ranking_key)
    return {
        "scope": list(RANKED_CANDIDATES),
        "predeclared_rule": list(RANKING_RULE),
        "ordered_candidates": ordered,
        "tiny_diagnostic_winner": ordered[0],
        "ranking_keys": {
            candidate: list(ranking_key(candidate)) for candidate in ordered
        },
        "interpretation": (
            "This ordering applies only to this fixed random tiny checkpoint and "
            "is not a LightEval result or a best-quantization claim."
        ),
    }


def _run_comparison(output_root: Path, artifact_path: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    canonical = output_root / "tiny-canonical-fp32"
    _create_canonical_checkpoint(canonical)

    checkpoints = {
        "fp16-baseline": output_root / "fp16-baseline",
        "bf16-baseline": output_root / "bf16-baseline",
        "nvfp4-w4a4": output_root / "nvfp4-w4a4",
        "nvfp4-w4a16": output_root / "nvfp4-w4a16",
    }
    _save_baseline(canonical, checkpoints["fp16-baseline"], torch.float16)
    _save_baseline(canonical, checkpoints["bf16-baseline"], torch.bfloat16)
    preparation = {
        "fp16-baseline": {"save_pretrained": True},
        "bf16-baseline": {"save_pretrained": True},
        "nvfp4-w4a4": _quantize_candidate(canonical, output_root, "nvfp4-w4a4"),
        "nvfp4-w4a16": _quantize_candidate(canonical, output_root, "nvfp4-w4a16"),
    }

    variants = []
    logits = {}
    with tempfile.TemporaryDirectory(
        prefix="rwkv7-tiny-comparison-", dir=output_root
    ) as temporary:
        temporary_root = Path(temporary)
        for variant in VARIANT_ORDER:
            result_path = temporary_root / f"{variant}.json"
            logits_path = temporary_root / f"{variant}.pt"
            process_tmp = temporary_root / f"{variant}-tmp"
            process_tmp.mkdir()
            _run_evaluator(
                checkpoints[variant],
                variant,
                result_path,
                logits_path,
                process_tmp,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["storage"] = _checkpoint_inventory(checkpoints[variant])
            result["preparation"] = preparation[variant]
            variants.append(result)
            logits[variant] = torch.load(logits_path, map_location="cpu")

    fp16_logits = logits["fp16-baseline"]
    bf16_logits = logits["bf16-baseline"]
    for variant in variants:
        variant_logits = logits[variant["variant"]]
        variant["quality"]["logits_error_vs_fp16"] = _logits_error(
            variant_logits, fp16_logits
        )
        variant["quality"]["logits_error_vs_bf16"] = _logits_error(
            variant_logits, bf16_logits
        )

    artifact = {
        "schema_version": 1,
        "scope": {
            "kind": "fixed-random-tiny-rwkv7-diagnostic",
            "formal_evaluation": False,
            "disallowed_interpretations": [
                "LightEval quality result",
                "trained-checkpoint accuracy result",
                "best quantization scheme claim",
            ],
        },
        "source": _git_revision(),
        "framework_versions": _framework_versions(),
        "workload": {
            "seed": SEED,
            "checkpoint": _tiny_config().to_dict(),
            "fixture_overrides": [
                (
                    "Set each TimeMix k_k tensor to one so the untrained tiny "
                    "fixture has a nonzero normalized-key path in FP16."
                )
            ],
            "prompt_ids": PROMPT_IDS,
            "quality_ids": QUALITY_IDS,
            "sampling": {
                "do_sample": True,
                "temperature": 0.8,
                "top_k": 8,
                "max_new_tokens": MAX_NEW_TOKENS,
                "use_cache": True,
            },
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
        },
        "measurement": {
            "included": (
                "standard Transformers generate prefill, recurrent-state decode, "
                "sampling, and CUDA synchronization"
            ),
            "excluded": (
                "checkpoint creation, quantization, save, reload, warmup, quality "
                "forward, artifact inventory, and JSON serialization"
            ),
            "process_isolation": "one fresh evaluator process per variant",
        },
        "variants": variants,
        "ranking": _rank_candidates(variants),
    }
    _atomic_json(artifact_path, artifact)
    print(json.dumps({"artifact": str(artifact_path), **artifact["ranking"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/rwkv7-tiny-candidate-comparison"),
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--evaluate-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANT_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--logits", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.evaluate_checkpoint:
        if not args.variant or not args.result or not args.logits:
            parser.error(
                "internal evaluator requires --variant, --result, and --logits"
            )
        _evaluate_checkpoint(
            args.evaluate_checkpoint, args.variant, args.result, args.logits
        )
        return

    artifact_path = args.artifact or args.output_root / "comparison.json"
    _run_comparison(args.output_root, artifact_path)


if __name__ == "__main__":
    main()
