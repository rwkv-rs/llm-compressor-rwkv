"""Formal evidence and Pareto selection for the RWKV-7 candidate search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llmcompressor.modifiers.quantization.rwkv7_evaluation import (
    _prompt_contract,
    build_rwkv7_pareto_artifact,
)

_CHECKPOINT_SHA256 = "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c"
_IMPLEMENTATION_REVISION = "a" * 40
_CANDIDATES = [
    "nvfp4-w4a4",
    "nvfp4-w4a16",
    "nvfp4-w4a16-protection-ablation",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> dict[str, str]:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"path": path.name, "sha256": _sha256(path)}


def _write_text(path: Path, value: str) -> dict[str, str]:
    path.write_text(value, encoding="utf-8")
    return {"path": path.name, "sha256": _sha256(path)}


def _audit(artifact_sha256: str) -> dict:
    return {
        "schema_version": 2,
        "weight_sha256": artifact_sha256,
        "wkv_mode": "fp32io16",
        "pool_manifest_lineage": None,
        "samples_per_task": 1,
        "tasks": {
            "gsm8k|0": [
                {
                    "document_index": 0,
                    "question": "2+2?",
                    "model_input_text": "User: 2+2?\nAssistant:",
                    "model_output_text": ["4"],
                    "scorer_input": {"golds": ["4"], "predictions": ["4"]},
                    "scorer_output": {"exact_match": 1.0},
                    "standard_answer": ["4"],
                }
            ]
        },
    }


def _artifact_result(candidate: str, artifact_sha256: str, size_bytes: int) -> dict:
    if candidate == "nvfp4-w4a16-protection-ablation":
        quantized_modules = [
            *(
                f"model.blocks.0.att.{name}"
                for name in ("w1", "w2", "a1", "a2", "g1", "g2")
            ),
            *(f"model.blocks.0.att.{name}" for name in ("receptance", "key", "output")),
            *(
                f"model.blocks.1.att.{name}"
                for name in (
                    "w1",
                    "w2",
                    "a1",
                    "a2",
                    "v1",
                    "v2",
                    "g1",
                    "g2",
                )
            ),
            *(
                f"model.blocks.1.att.{name}"
                for name in ("receptance", "key", "value", "output")
            ),
        ]
    else:
        quantized_modules = [
            f"model.blocks.{layer}.ffn.{name}"
            for layer in range(2)
            for name in ("key", "value")
        ]
    consumer_capability = (
        "vllm-rwkv-nvfp4-w4a4" if candidate == "nvfp4-w4a4" else "vllm-rwkv-nvfp4-w4a16"
    )
    target_schema = (
        "rwkv7-nvfp4-protection-ablation-no-ffn-v1"
        if candidate == "nvfp4-w4a16-protection-ablation"
        else "rwkv7-nvfp4-critical-high-v1"
    )
    target_digest = hashlib.sha256(
        json.dumps(
            quantized_modules,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "implementation_revision": _IMPLEMENTATION_REVISION,
        "checkpoint": {"sha256": _CHECKPOINT_SHA256},
        "candidate": candidate,
        "execution": {
            "fresh_reload": {"passed": True},
            "artifact_contract": {
                "vllm": {
                    "quantization_format": "nvfp4-pack-quantized",
                    "target_schema_version": 1,
                    "target_schema": target_schema,
                    "num_hidden_layers": 2,
                    "quantized_target_fqns_digest": target_digest,
                    "consumer_capabilities": [
                        "transformers-rwkv-compressed-tensors",
                        consumer_capability,
                    ],
                    "vllm_consumer_requirement": consumer_capability,
                    "vllm_consumer_revision": "c" * 40,
                    "quantized_modules": quantized_modules,
                    "protected_modules": ["model.blocks.0.att.value"],
                }
            },
        },
        "candidate_artifact": {
            "sha256": artifact_sha256,
            "size_bytes": size_bytes,
        },
        "standard_checkpoint": {
            "sha256": "1" * 64,
            "size_bytes": 3000,
        },
        "formal_checkpoint": True,
        "formal_evaluation": False,
        "diagnostic_tiny": False,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    lighteval_config = _write_text(
        tmp_path / "lighteval.toml",
        'schema_version = 1\nwkv_modes = ["fp32io16"]\n',
    )
    performance_workload = _write_text(
        tmp_path / "performance.json",
        '{"batch":320,"decode":1280,"prompt":128}\n',
    )
    evalscope_config = _write_text(
        tmp_path / "evalscope.yaml",
        "datasets:\n  - gsm8k\n",
    )
    prompt_sha256, _ = _prompt_contract(_audit("1" * 64))
    contract = {
        "schema_version": 1,
        "implementation_revision": _IMPLEMENTATION_REVISION,
        "checkpoint_sha256": _CHECKPOINT_SHA256,
        "candidate_order": _CANDIDATES,
        "lighteval": {
            "pipeline_revision": "b" * 40,
            "config_sha256": lighteval_config["sha256"],
            "task_names": ["gsm8k|0"],
            "prompt_contract_sha256": prompt_sha256,
            "wkv_mode": "fp32io16",
            "quality_metrics": {
                "gsm8k/exact_match": {
                    "direction": "higher",
                    "max_regression": 0.25,
                    "weight": 1.0,
                }
            },
        },
        "performance": {
            "pipeline_revision": "c" * 40,
            "workload_sha256": performance_workload["sha256"],
            "consumer_capabilities": [
                "vllm-rwkv-nvfp4-w4a4",
                "vllm-rwkv-nvfp4-w4a16",
            ],
            "wkv_mode": "fp16",
            "gemm_accumulation_policy": "fp16",
            "measurement_scope": "canonical-vllm-rwkv",
        },
        "evalscope": {
            "pipeline_revision": "d" * 40,
            "config_sha256": evalscope_config["sha256"],
            "evaluation_contract_sha256": "e" * 64,
        },
        "selection_priority": [
            "quality_delta_score",
            "throughput_tokens_per_second",
            "artifact_size_bytes",
            "peak_vram_bytes",
            "latency_p50_ms",
        ],
    }
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)

    candidate_values = {
        "nvfp4-w4a4": ("2" * 64, 100, 0.80, 200, 20.0, 50.0, 999.0),
        "nvfp4-w4a16": ("3" * 64, 200, 0.95, 220, 15.0, 70.0, 1.0),
        "nvfp4-w4a16-protection-ablation": (
            "4" * 64,
            180,
            0.90,
            210,
            14.0,
            72.0,
            2.0,
        ),
    }
    artifact_refs = {}
    for candidate, values in candidate_values.items():
        artifact_sha256, size_bytes, *_ = values
        artifact_refs[candidate] = _write_json(
            tmp_path / f"{candidate}-artifact.json",
            _artifact_result(candidate, artifact_sha256, size_bytes),
        )
    baseline_artifact_ref = _write_json(
        tmp_path / "baseline-artifact.json",
        _artifact_result("nvfp4-w4a4", "0" * 64, 3000),
    )

    variants = []
    variant_values = {
        "baseline-bf16": ("1" * 64, 1.0, 300, 30.0, 40.0, 0.0),
        **{
            candidate: (
                values[0],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
            )
            for candidate, values in candidate_values.items()
        },
    }
    for variant, values in variant_values.items():
        artifact_sha256, quality, vram, latency, throughput, evalscope_score = values
        result_ref = (
            baseline_artifact_ref
            if variant == "baseline-bf16"
            else artifact_refs[variant]
        )
        lighteval_result = _write_json(
            tmp_path / f"{variant}-lighteval.json",
            {
                "schema_version": 2,
                "weight_sha256": artifact_sha256,
                "wkv_mode": "fp32io16",
                "pool_manifest_lineage": None,
                "metrics": {"gsm8k/exact_match": quality},
            },
        )
        audit_ref = _write_json(
            tmp_path / f"{variant}-audit.json",
            _audit(artifact_sha256),
        )
        performance_ref = _write_json(
            tmp_path / f"{variant}-performance.json",
            {
                "schema_version": 1,
                "artifact_sha256": artifact_sha256,
                "checkpoint_sha256": _CHECKPOINT_SHA256,
                "pipeline_revision": "c" * 40,
                "workload_sha256": performance_workload["sha256"],
                "measurement_scope": "canonical-vllm-rwkv",
                "canonical_performance_acceptance": True,
                "wkv_mode": "fp16",
                "gemm_accumulation_policy": "fp16",
                "peak_vram_bytes": vram,
                "latency_p50_ms": latency,
                "throughput_tokens_per_second": throughput,
                "device_name": "NVIDIA RTX PRO 6000 Blackwell",
            },
        )
        evalscope_ref = _write_json(
            tmp_path / f"{variant}-evalscope.json",
            {
                "schema_version": 1,
                "evaluator": "evalscope",
                "artifact_sha256": artifact_sha256,
                "pipeline_revision": "d" * 40,
                "config_sha256": evalscope_config["sha256"],
                "evaluation_contract_sha256": "e" * 64,
                "metrics": {"gsm8k/accuracy": evalscope_score},
            },
        )
        variants.append(
            {
                "variant": variant,
                "role": "baseline" if variant == "baseline-bf16" else "candidate",
                "artifact_result": result_ref,
                "lighteval_result": lighteval_result,
                "lighteval_sample_audit": audit_ref,
                "performance_result": performance_ref,
                "evalscope_result": evalscope_ref,
            }
        )
    evidence = {
        "schema_version": 1,
        "lighteval_config": lighteval_config,
        "performance_workload": performance_workload,
        "evalscope_config": evalscope_config,
        "variants": variants,
    }
    evidence_path = tmp_path / "evidence.json"
    _write_json(evidence_path, evidence)
    return contract_path, evidence_path, tmp_path / "report.json", evidence


@pytest.mark.unit
def test_builds_traceable_formal_pareto_without_using_evalscope_for_selection(
    tmp_path,
):
    contract_path, evidence_path, output_path, _ = _fixture(tmp_path)

    report = build_rwkv7_pareto_artifact(
        contract_path,
        evidence_path,
        output_path,
    )

    assert report["formal_checkpoint"] is True
    assert report["formal_evaluation"] is True
    assert report["diagnostic_tiny"] is False
    assert report["pareto"]["selected_candidate"] == ("nvfp4-w4a16")
    assert report["pareto"]["evalscope_used_for_selection"] is False
    assert report["pareto"]["quality_eligible"] == _CANDIDATES
    assert [row["variant"] for row in report["variants"]] == [
        "baseline-bf16",
        *_CANDIDATES,
    ]
    assert all(
        row["evalscope"]["selection_role"] == "cross-check-only"
        for row in report["variants"]
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


@pytest.mark.unit
def test_rejects_an_incomplete_candidate_search(tmp_path):
    contract_path, evidence_path, output_path, evidence = _fixture(tmp_path)
    evidence["variants"].pop()
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="baseline plus every closed candidate"):
        build_rwkv7_pareto_artifact(contract_path, evidence_path, output_path)


@pytest.mark.unit
def test_rejects_w8_diagnostic_in_vllm_comparison_contract(tmp_path):
    contract_path, evidence_path, output_path, _ = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["candidate_order"].append("w8a16-low-rank-critical-high")
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="closed candidate order"):
        build_rwkv7_pareto_artifact(contract_path, evidence_path, output_path)


@pytest.mark.unit
def test_rejects_comparison_without_exact_w4a4_and_w4a16_capabilities(tmp_path):
    contract_path, evidence_path, output_path, _ = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["performance"]["consumer_capabilities"] = ["vllm-rwkv-nvfp4-w4a16"]
    _write_json(contract_path, contract)

    with pytest.raises(ValueError, match="exact W4A4 and W4A16"):
        build_rwkv7_pareto_artifact(contract_path, evidence_path, output_path)


@pytest.mark.unit
def test_rejects_non_vllm_consumer_from_canonical_publication(tmp_path):
    contract_path, evidence_path, output_path, evidence = _fixture(tmp_path)
    variant = evidence["variants"][1]
    artifact_path = tmp_path / variant["artifact_result"]["path"]
    artifact_result = json.loads(artifact_path.read_text(encoding="utf-8"))
    loader_contract = artifact_result["execution"]["artifact_contract"]["vllm"]
    loader_contract["consumer_capabilities"] = ["transformers-rwkv-compressed-tensors"]
    loader_contract["vllm_consumer_revision"] = None
    variant["artifact_result"] = _write_json(artifact_path, artifact_result)
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="lacks its exact executable vLLM RWKV"):
        build_rwkv7_pareto_artifact(contract_path, evidence_path, output_path)


@pytest.mark.unit
def test_rejects_protection_ablation_target_drift_from_vllm_matrix(tmp_path):
    contract_path, evidence_path, output_path, evidence = _fixture(tmp_path)
    variant = next(
        row
        for row in evidence["variants"]
        if row["variant"] == "nvfp4-w4a16-protection-ablation"
    )
    artifact_path = tmp_path / variant["artifact_result"]["path"]
    artifact_result = json.loads(artifact_path.read_text(encoding="utf-8"))
    loader_contract = artifact_result["execution"]["artifact_contract"]["vllm"]
    loader_contract["quantized_modules"].pop()
    loader_contract["quantized_target_fqns_digest"] = hashlib.sha256(
        json.dumps(
            loader_contract["quantized_modules"],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    variant["artifact_result"] = _write_json(artifact_path, artifact_result)
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match=r"no-FFN 9\+12\*\(L-1\)"):
        build_rwkv7_pareto_artifact(contract_path, evidence_path, output_path)


@pytest.mark.unit
def test_rejects_quantized_target_digest_tamper(tmp_path):
    contract_path, evidence_path, output_path, evidence = _fixture(tmp_path)
    variant = evidence["variants"][2]
    artifact_path = tmp_path / variant["artifact_result"]["path"]
    artifact_result = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_result["execution"]["artifact_contract"]["vllm"][
        "quantized_target_fqns_digest"
    ] = "0" * 64
    variant["artifact_result"] = _write_json(artifact_path, artifact_result)
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="invalid exact target schema or digest"):
        build_rwkv7_pareto_artifact(contract_path, evidence_path, output_path)


@pytest.mark.unit
def test_rejects_lighteval_prompt_drift_even_with_a_rebound_file(tmp_path):
    contract_path, evidence_path, output_path, evidence = _fixture(tmp_path)
    audit_ref = evidence["variants"][2]["lighteval_sample_audit"]
    audit_path = tmp_path / audit_ref["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["tasks"]["gsm8k|0"][0]["model_input_text"] = "different prompt"
    evidence["variants"][2]["lighteval_sample_audit"] = _write_json(audit_path, audit)
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="LightEval prompt contract mismatch"):
        build_rwkv7_pareto_artifact(contract_path, evidence_path, output_path)


@pytest.mark.unit
def test_rejects_fresh_process_diagnostic_as_canonical_performance(tmp_path):
    contract_path, evidence_path, output_path, evidence = _fixture(tmp_path)
    performance_ref = evidence["variants"][1]["performance_result"]
    performance_path = tmp_path / performance_ref["path"]
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["measurement_scope"] = "fresh-process-transformers-generate-diagnostic"
    performance["canonical_performance_acceptance"] = False
    evidence["variants"][1]["performance_result"] = _write_json(
        performance_path, performance
    )
    _write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="validation errors"):
        build_rwkv7_pareto_artifact(contract_path, evidence_path, output_path)
