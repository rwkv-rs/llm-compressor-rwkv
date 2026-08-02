"""Traceable formal comparison for the closed RWKV-7 quantization search."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmcompressor.modifiers.quantization.rwkv7 import (
    _TRANSFORMERS_RWKV_CONSUMER,
    _VLLM_RWKV_NVFP4_W4A4_CONSUMER,
    _VLLM_RWKV_NVFP4_W4A16_CONSUMER,
    RWKV7CheckpointContract,
    _atomic_json,
    _sha256_file,
    _target_fqns_digest,
    _vllm_nvfp4_protection_ablation_targets,
)

__all__ = [
    "RWKV7ComparisonContract",
    "RWKV7ComparisonEvidence",
    "build_rwkv7_pareto_artifact",
]


_CANDIDATES = [
    "nvfp4-w4a4",
    "nvfp4-w4a16",
    "nvfp4-w4a16-protection-ablation",
]
_VARIANTS = ["baseline-bf16", *_CANDIDATES]
_SELECTION_METRICS = {
    "quality_delta_score": "max",
    "throughput_tokens_per_second": "max",
    "artifact_size_bytes": "min",
    "peak_vram_bytes": "min",
    "latency_p50_ms": "min",
}
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_OID_PATTERN = r"^[0-9a-f]{40}$"


class EvidenceFile(BaseModel):
    """A source file whose content is bound before report construction."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class QualityMetricRule(BaseModel):
    """A predeclared LightEval quality gate and aggregate weight."""

    model_config = ConfigDict(extra="forbid")

    direction: Literal["higher", "lower"]
    max_regression: float = Field(ge=0.0)
    weight: float = Field(gt=0.0)


class LightEvalComparisonContract(BaseModel):
    """Identity shared by baseline and every quantized LightEval run."""

    model_config = ConfigDict(extra="forbid")

    pipeline_revision: str = Field(pattern=_GIT_OID_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_names: list[str]
    prompt_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    wkv_mode: Literal["fp32io16"] = "fp32io16"
    quality_metrics: dict[str, QualityMetricRule]

    @model_validator(mode="after")
    def validate_tasks_and_metrics(self):
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("LightEval task_names must be non-empty and unique")
        if any(not name for name in self.task_names):
            raise ValueError("LightEval task_names cannot contain an empty name")
        if not self.quality_metrics:
            raise ValueError("LightEval quality_metrics cannot be empty")
        return self


class PerformanceComparisonContract(BaseModel):
    """Canonical serving workload shared by every compared artifact."""

    model_config = ConfigDict(extra="forbid")

    pipeline_revision: str = Field(pattern=_GIT_OID_PATTERN)
    workload_sha256: str = Field(pattern=_SHA256_PATTERN)
    consumer_capabilities: list[
        Literal[
            "vllm-rwkv-nvfp4-w4a4",
            "vllm-rwkv-nvfp4-w4a16",
        ]
    ]
    wkv_mode: Literal["fp16"] = "fp16"
    gemm_accumulation_policy: Literal["fp16"] = "fp16"
    measurement_scope: Literal["canonical-vllm-rwkv"] = "canonical-vllm-rwkv"

    @model_validator(mode="after")
    def validate_exact_consumer_capabilities(self):
        if self.consumer_capabilities != [
            _VLLM_RWKV_NVFP4_W4A4_CONSUMER,
            _VLLM_RWKV_NVFP4_W4A16_CONSUMER,
        ]:
            raise ValueError(
                "canonical vLLM-RWKV comparison requires exact W4A4 and W4A16 "
                "consumer capabilities"
            )
        return self


class EvalScopeComparisonContract(BaseModel):
    """EvalScope identity retained as a separate cross-check."""

    model_config = ConfigDict(extra="forbid")

    pipeline_revision: str = Field(pattern=_GIT_OID_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluation_contract_sha256: str = Field(pattern=_SHA256_PATTERN)


class RWKV7ComparisonContract(BaseModel):
    """Predeclared rules for one complete real-checkpoint search."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    implementation_revision: str = Field(pattern=_GIT_OID_PATTERN)
    checkpoint_sha256: Literal[
        "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c"
    ] = RWKV7CheckpointContract().sha256
    candidate_order: list[str]
    lighteval: LightEvalComparisonContract
    performance: PerformanceComparisonContract
    evalscope: EvalScopeComparisonContract
    selection_priority: list[str]

    @model_validator(mode="after")
    def validate_closed_search(self):
        if self.candidate_order != _CANDIDATES:
            raise ValueError("RWKV-7 comparison requires the closed candidate order")
        if (
            not self.selection_priority
            or len(set(self.selection_priority)) != len(self.selection_priority)
            or any(
                metric not in _SELECTION_METRICS for metric in self.selection_priority
            )
        ):
            raise ValueError("RWKV-7 selection_priority is invalid")
        return self


class VariantEvidence(BaseModel):
    """Files produced for one baseline or quantized artifact."""

    model_config = ConfigDict(extra="forbid")

    variant: str
    role: Literal["baseline", "candidate"]
    artifact_result: EvidenceFile
    lighteval_result: EvidenceFile
    lighteval_sample_audit: EvidenceFile
    performance_result: EvidenceFile
    evalscope_result: EvidenceFile


class RWKV7ComparisonEvidence(BaseModel):
    """Complete file inventory for the predeclared search."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    lighteval_config: EvidenceFile
    performance_workload: EvidenceFile
    evalscope_config: EvidenceFile
    variants: list[VariantEvidence]

    @model_validator(mode="after")
    def validate_complete_search(self):
        names = [variant.variant for variant in self.variants]
        if names != _VARIANTS:
            raise ValueError(
                "RWKV-7 evidence requires baseline plus every closed candidate"
            )
        expected_roles = ["baseline", *(["candidate"] * len(_CANDIDATES))]
        if [variant.role for variant in self.variants] != expected_roles:
            raise ValueError("RWKV-7 evidence variant roles are invalid")
        return self


class CanonicalPerformanceEvidence(BaseModel):
    """Normalized output of the canonical vLLM-RWKV performance run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    checkpoint_sha256: Literal[
        "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c"
    ] = RWKV7CheckpointContract().sha256
    pipeline_revision: str = Field(pattern=_GIT_OID_PATTERN)
    workload_sha256: str = Field(pattern=_SHA256_PATTERN)
    measurement_scope: Literal["canonical-vllm-rwkv"]
    canonical_performance_acceptance: Literal[True]
    wkv_mode: Literal["fp16"]
    gemm_accumulation_policy: Literal["fp16"]
    peak_vram_bytes: int = Field(gt=0)
    latency_p50_ms: float = Field(gt=0.0)
    throughput_tokens_per_second: float = Field(gt=0.0)
    device_name: str = Field(min_length=1)


class EvalScopeEvidence(BaseModel):
    """Normalized EvalScope result; it never participates in selection."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    evaluator: Literal["evalscope"]
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    pipeline_revision: str = Field(pattern=_GIT_OID_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluation_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    metrics: dict[str, float]

    @model_validator(mode="after")
    def validate_metrics(self):
        _finite_metrics(self.metrics, "EvalScope")
        return self


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} does not exist: {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _read_evidence_file(
    reference: EvidenceFile,
    *,
    base_dir: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(reference.path)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != reference.sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: "
            f"expected={reference.sha256} actual={actual_sha256}"
        )
    return _read_json(path, label), {
        "path": reference.path,
        "sha256": actual_sha256,
    }


def _verify_evidence_file(
    reference: EvidenceFile,
    *,
    base_dir: Path,
    expected_sha256: str,
    label: str,
) -> dict[str, str]:
    path = Path(reference.path)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != reference.sha256 or actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: "
            f"contract={expected_sha256} evidence={reference.sha256} "
            f"actual={actual_sha256}"
        )
    return {"path": reference.path, "sha256": actual_sha256}


def _finite_metrics(metrics: Any, label: str) -> dict[str, float]:
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"{label} metrics must be a non-empty object")
    normalized = {}
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} metrics contain an invalid value")
        normalized[name] = float(value)
    return normalized


def _prompt_contract(audit: dict[str, Any]) -> tuple[str, list[str]]:
    if audit.get("schema_version") != 2:
        raise ValueError("LightEval sample audit schema_version must be 2")
    tasks = audit.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("LightEval sample audit lacks tasks")
    prompt_rows = {}
    for task_name, rows in tasks.items():
        if not isinstance(task_name, str) or not isinstance(rows, list) or not rows:
            raise ValueError("LightEval sample audit has an invalid task")
        normalized_rows = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("LightEval sample audit row must be an object")
            try:
                normalized_rows.append(
                    {
                        "document_index": row["document_index"],
                        "question": row["question"],
                        "model_input_text": row["model_input_text"],
                        "standard_answer": row["standard_answer"],
                    }
                )
            except KeyError as error:
                raise ValueError(
                    "LightEval sample audit lacks prompt identity fields"
                ) from error
        prompt_rows[task_name] = normalized_rows
    return _canonical_digest(prompt_rows), list(tasks)


def _artifact_identity(
    *,
    variant: VariantEvidence,
    result: dict[str, Any],
    contract: RWKV7ComparisonContract,
) -> dict[str, Any]:
    if (
        result.get("schema_version") != 1
        or result.get("implementation_revision") != contract.implementation_revision
        or result.get("formal_checkpoint") is not True
        or result.get("formal_evaluation") is not False
        or result.get("diagnostic_tiny") is not False
        or result.get("checkpoint", {}).get("sha256") != contract.checkpoint_sha256
    ):
        raise ValueError(
            f"{variant.variant} artifact result is not formal 1.5B evidence"
        )
    if variant.role == "baseline":
        artifact = result.get("standard_checkpoint")
    else:
        if result.get("candidate") != variant.variant:
            raise ValueError(f"{variant.variant} artifact result candidate mismatch")
        execution = result.get("execution", {})
        artifact_contract = execution.get("artifact_contract", {})
        loader_contract = artifact_contract.get("vllm", {})
        protected_modules = loader_contract.get("protected_modules", [])
        if (
            execution.get("fresh_reload", {}).get("passed") is not True
            or "model.blocks.0.att.value" not in protected_modules
        ):
            raise ValueError(
                f"{variant.variant} lacks fresh reload or layer-0 v_first protection"
            )
        expected_consumer_capability = (
            _VLLM_RWKV_NVFP4_W4A4_CONSUMER
            if variant.variant == "nvfp4-w4a4"
            else _VLLM_RWKV_NVFP4_W4A16_CONSUMER
        )
        if (
            loader_contract.get("quantization_format") != "nvfp4-pack-quantized"
            or loader_contract.get("vllm_consumer_requirement")
            != expected_consumer_capability
            or loader_contract.get("consumer_capabilities")
            != [_TRANSFORMERS_RWKV_CONSUMER, expected_consumer_capability]
            or expected_consumer_capability
            not in contract.performance.consumer_capabilities
            or loader_contract.get("vllm_consumer_revision")
            != contract.performance.pipeline_revision
        ):
            raise ValueError(
                f"{variant.variant} lacks its exact executable vLLM RWKV "
                "consumer capability"
            )
        quantized_modules = loader_contract.get("quantized_modules")
        if (
            loader_contract.get("target_schema_version") != 1
            or not isinstance(quantized_modules, list)
            or loader_contract.get("quantized_target_fqns_digest")
            != _target_fqns_digest(quantized_modules)
        ):
            raise ValueError(
                f"{variant.variant} has an invalid exact target schema or digest"
            )
        if variant.variant == "nvfp4-w4a16-protection-ablation":
            num_hidden_layers = loader_contract.get("num_hidden_layers")
            if not isinstance(num_hidden_layers, int) or num_hidden_layers < 2:
                raise ValueError(
                    "nvfp4-w4a16-protection-ablation has an invalid layer count"
                )
            expected_targets = _vllm_nvfp4_protection_ablation_targets(
                "model",
                num_hidden_layers,
            )
            if (
                loader_contract.get("target_schema")
                != "rwkv7-nvfp4-protection-ablation-no-ffn-v1"
                or quantized_modules != expected_targets
                or len(expected_targets) != 9 + 12 * (num_hidden_layers - 1)
                or any(".ffn." in name for name in expected_targets)
            ):
                raise ValueError(
                    "nvfp4-w4a16-protection-ablation does not match the exact "
                    "no-FFN 9+12*(L-1) vLLM consumer matrix"
                )
        elif loader_contract.get("target_schema") != "rwkv7-nvfp4-critical-high-v1":
            raise ValueError(f"{variant.variant} has an invalid target schema")
        artifact = result.get("candidate_artifact")
    if (
        not isinstance(artifact, dict)
        or not isinstance(artifact.get("size_bytes"), int)
        or artifact["size_bytes"] <= 0
        or not isinstance(artifact.get("sha256"), str)
        or not re.fullmatch(_SHA256_PATTERN, artifact["sha256"])
    ):
        raise ValueError(f"{variant.variant} has an invalid artifact manifest")
    return {
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
    }


def _lighteval_identity(
    *,
    variant: str,
    artifact_sha256: str,
    result: dict[str, Any],
    audit: dict[str, Any],
    contract: LightEvalComparisonContract,
) -> dict[str, Any]:
    if result.get("schema_version") != 2:
        raise ValueError(f"{variant} LightEval result schema_version must be 2")
    if (
        result.get("weight_sha256") != artifact_sha256
        or audit.get("weight_sha256") != artifact_sha256
    ):
        raise ValueError(f"{variant} LightEval result is bound to another artifact")
    if (
        result.get("wkv_mode") != contract.wkv_mode
        or audit.get("wkv_mode") != contract.wkv_mode
    ):
        raise ValueError(f"{variant} LightEval WKV mode mismatch")
    metrics = _finite_metrics(result.get("metrics"), f"{variant} LightEval")
    if set(metrics) != set(contract.quality_metrics):
        raise ValueError(f"{variant} LightEval metric set differs from quality gates")
    prompt_sha256, task_names = _prompt_contract(audit)
    if task_names != contract.task_names:
        raise ValueError(f"{variant} LightEval task order mismatch")
    if prompt_sha256 != contract.prompt_contract_sha256:
        raise ValueError(f"{variant} LightEval prompt contract mismatch")
    return {
        "metrics": metrics,
        "prompt_contract_sha256": prompt_sha256,
        "task_names": task_names,
        "wkv_mode": contract.wkv_mode,
        "pool_manifest_lineage": result.get("pool_manifest_lineage"),
    }


def _quality_row(
    baseline: dict[str, float],
    candidate: dict[str, float],
    rules: dict[str, QualityMetricRule],
) -> dict[str, Any]:
    regressions = {}
    oriented_deltas = {}
    weighted_delta = 0.0
    total_weight = 0.0
    for metric, rule in rules.items():
        if rule.direction == "higher":
            regression = baseline[metric] - candidate[metric]
            oriented_delta = candidate[metric] - baseline[metric]
        else:
            regression = candidate[metric] - baseline[metric]
            oriented_delta = baseline[metric] - candidate[metric]
        regressions[metric] = regression
        oriented_deltas[metric] = oriented_delta
        weighted_delta += oriented_delta * rule.weight
        total_weight += rule.weight
    return {
        "eligible": all(
            regressions[metric] <= rule.max_regression for metric, rule in rules.items()
        ),
        "regressions": regressions,
        "oriented_deltas": oriented_deltas,
        "quality_delta_score": weighted_delta / total_weight,
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    comparisons = [
        left["quality_delta_score"] >= right["quality_delta_score"],
        left["throughput_tokens_per_second"] >= right["throughput_tokens_per_second"],
        left["artifact_size_bytes"] <= right["artifact_size_bytes"],
        left["peak_vram_bytes"] <= right["peak_vram_bytes"],
        left["latency_p50_ms"] <= right["latency_p50_ms"],
    ]
    strict = [
        left["quality_delta_score"] > right["quality_delta_score"],
        left["throughput_tokens_per_second"] > right["throughput_tokens_per_second"],
        left["artifact_size_bytes"] < right["artifact_size_bytes"],
        left["peak_vram_bytes"] < right["peak_vram_bytes"],
        left["latency_p50_ms"] < right["latency_p50_ms"],
    ]
    return all(comparisons) and any(strict)


def _select_candidate(
    rows: list[dict[str, Any]],
    priority: list[str],
) -> tuple[list[str], str]:
    eligible = [row for row in rows if row["quality_eligible"]]
    if not eligible:
        raise ValueError("no RWKV-7 candidate satisfies the predeclared quality gates")
    frontier = [
        row
        for row in eligible
        if not any(other is not row and _dominates(other, row) for other in eligible)
    ]

    def rank(row: dict[str, Any]):
        values = []
        for metric in priority:
            value = row[metric]
            values.append(-value if _SELECTION_METRICS[metric] == "max" else value)
        values.append(_CANDIDATES.index(row["variant"]))
        return tuple(values)

    selected = min(frontier, key=rank)
    return [row["variant"] for row in frontier], selected["variant"]


def build_rwkv7_pareto_artifact(
    contract_path: Path,
    evidence_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate all formal evidence and emit one selection-ready JSON artifact."""

    contract_path = contract_path.resolve()
    evidence_path = evidence_path.resolve()
    contract_payload = _read_json(contract_path, "RWKV-7 comparison contract")
    evidence_payload = _read_json(evidence_path, "RWKV-7 comparison evidence")
    contract = RWKV7ComparisonContract.model_validate(contract_payload)
    evidence = RWKV7ComparisonEvidence.model_validate(evidence_payload)
    contract_sources = {
        "lighteval_config": _verify_evidence_file(
            evidence.lighteval_config,
            base_dir=evidence_path.parent,
            expected_sha256=contract.lighteval.config_sha256,
            label="LightEval config",
        ),
        "performance_workload": _verify_evidence_file(
            evidence.performance_workload,
            base_dir=evidence_path.parent,
            expected_sha256=contract.performance.workload_sha256,
            label="performance workload",
        ),
        "evalscope_config": _verify_evidence_file(
            evidence.evalscope_config,
            base_dir=evidence_path.parent,
            expected_sha256=contract.evalscope.config_sha256,
            label="EvalScope config",
        ),
    }
    rows = []
    traces = []
    baseline_metrics = None

    for variant in evidence.variants:
        sources = {}
        artifact_result, sources["artifact_result"] = _read_evidence_file(
            variant.artifact_result,
            base_dir=evidence_path.parent,
            label=f"{variant.variant} artifact result",
        )
        artifact = _artifact_identity(
            variant=variant,
            result=artifact_result,
            contract=contract,
        )
        lighteval_result, sources["lighteval_result"] = _read_evidence_file(
            variant.lighteval_result,
            base_dir=evidence_path.parent,
            label=f"{variant.variant} LightEval result",
        )
        lighteval_audit, sources["lighteval_sample_audit"] = _read_evidence_file(
            variant.lighteval_sample_audit,
            base_dir=evidence_path.parent,
            label=f"{variant.variant} LightEval sample audit",
        )
        lighteval = _lighteval_identity(
            variant=variant.variant,
            artifact_sha256=artifact["sha256"],
            result=lighteval_result,
            audit=lighteval_audit,
            contract=contract.lighteval,
        )
        performance_payload, sources["performance_result"] = _read_evidence_file(
            variant.performance_result,
            base_dir=evidence_path.parent,
            label=f"{variant.variant} performance result",
        )
        performance = CanonicalPerformanceEvidence.model_validate(performance_payload)
        if (
            performance.artifact_sha256 != artifact["sha256"]
            or performance.checkpoint_sha256 != contract.checkpoint_sha256
            or performance.pipeline_revision != contract.performance.pipeline_revision
            or performance.workload_sha256 != contract.performance.workload_sha256
            or performance.measurement_scope != contract.performance.measurement_scope
            or performance.wkv_mode != contract.performance.wkv_mode
            or performance.gemm_accumulation_policy
            != contract.performance.gemm_accumulation_policy
        ):
            raise ValueError(f"{variant.variant} performance contract mismatch")
        evalscope_payload, sources["evalscope_result"] = _read_evidence_file(
            variant.evalscope_result,
            base_dir=evidence_path.parent,
            label=f"{variant.variant} EvalScope result",
        )
        evalscope = EvalScopeEvidence.model_validate(evalscope_payload)
        if (
            evalscope.artifact_sha256 != artifact["sha256"]
            or evalscope.pipeline_revision != contract.evalscope.pipeline_revision
            or evalscope.config_sha256 != contract.evalscope.config_sha256
            or evalscope.evaluation_contract_sha256
            != contract.evalscope.evaluation_contract_sha256
        ):
            raise ValueError(f"{variant.variant} EvalScope contract mismatch")

        if variant.role == "baseline":
            baseline_metrics = lighteval["metrics"]
            quality = {
                "eligible": True,
                "regressions": {metric: 0.0 for metric in baseline_metrics},
                "oriented_deltas": {metric: 0.0 for metric in baseline_metrics},
                "quality_delta_score": 0.0,
            }
        else:
            if baseline_metrics is None:
                raise RuntimeError("baseline evidence must be processed first")
            quality = _quality_row(
                baseline_metrics,
                lighteval["metrics"],
                contract.lighteval.quality_metrics,
            )
        row = {
            "variant": variant.variant,
            "role": variant.role,
            "artifact": artifact,
            "performance": performance.model_dump(mode="json"),
            "lighteval": lighteval,
            "evalscope": {
                **evalscope.model_dump(mode="json"),
                "selection_role": "cross-check-only",
            },
            "quality": quality,
        }
        rows.append(row)
        traces.append({"variant": variant.variant, "sources": sources})

    candidate_rows = [
        {
            "variant": row["variant"],
            "quality_eligible": row["quality"]["eligible"],
            "quality_delta_score": row["quality"]["quality_delta_score"],
            "artifact_size_bytes": row["artifact"]["size_bytes"],
            "peak_vram_bytes": row["performance"]["peak_vram_bytes"],
            "latency_p50_ms": row["performance"]["latency_p50_ms"],
            "throughput_tokens_per_second": row["performance"][
                "throughput_tokens_per_second"
            ],
        }
        for row in rows
        if row["role"] == "candidate"
    ]
    frontier, selected = _select_candidate(candidate_rows, contract.selection_priority)
    report = {
        "schema_version": 1,
        "style": "unsloth-traceable-quantization-v1",
        "formal_checkpoint": True,
        "formal_evaluation": True,
        "diagnostic_tiny": False,
        "implementation_revision": contract.implementation_revision,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "comparison_contract": contract.model_dump(mode="json"),
        "comparison_contract_sha256": _canonical_digest(contract_payload),
        "evidence_manifest_sha256": _sha256_file(evidence_path),
        "variants": rows,
        "pareto": {
            "objectives": _SELECTION_METRICS,
            "quality_eligible": [
                row["variant"] for row in candidate_rows if row["quality_eligible"]
            ],
            "frontier": frontier,
            "selection_priority": contract.selection_priority,
            "selected_candidate": selected,
            "evalscope_used_for_selection": False,
        },
        "traceability": {
            "contract_sources": contract_sources,
            "variant_sources": traces,
        },
    }
    _atomic_json(output_path.resolve(), report)
    return report
