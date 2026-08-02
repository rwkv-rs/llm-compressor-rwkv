"""Conservative quantization targeting for standard Transformers RWKV-7."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "QuantizationTargetPolicyDecision",
    "QuantizationTargetPolicyMetadata",
    "RWKV7ArtifactContract",
    "RWKV7CheckpointContract",
    "RWKV7LowRankW8Metadata",
    "RWKV7QuantizationRecipeMetadata",
    "RWKV7RepositoryContract",
    "apply_rwkv7_target_policy",
    "build_rwkv7_artifact_contract",
    "build_rwkv7_quantization_recipe",
    "quantize_rwkv7_oneshot",
    "audit_rwkv7_quantized_checkpoint",
    "run_rwkv7_checkpoint_candidate",
    "validate_rwkv7_low_rank_standard_load",
    "verify_rwkv7_checkpoint",
]


_HEAD_IGNORE = "head"
_TIME_MIX_LINEAR_NAMES = ("receptance", "key", "value", "output")
_TIME_MIX_PARAMETER_NAMES = (
    "x_r",
    "x_w",
    "x_k",
    "x_v",
    "x_a",
    "x_g",
    "w0",
    "w1",
    "w2",
    "a0",
    "a1",
    "a2",
    "g1",
    "g2",
    "k_k",
    "k_a",
    "r_k",
)
_LOW_RANK_W8_PARAMETER_NAMES = ("w1", "w2", "a1", "a2", "g1", "g2")
_LOW_RANK_W8_GROUP_SIZE = 128
_LOW_RANK_W8_SIDECAR = "rwkv7_low_rank_w8.safetensors"
_SUPPORTED_FRAMEWORK_VERSIONS = {
    "compressed_tensors": "0.17.2.a20260731",
    "transformers": "5.15.0.dev0",
}
_CANDIDATE_SPECS = {
    "nvfp4-w4a4": {
        "scheme": "NVFP4",
        "protection_profile": "critical-high",
    },
    "nvfp4-w4a16": {
        "scheme": "NVFP4A16",
        "protection_profile": "critical-high",
    },
    "nvfp4-w4a16-protection-ablation": {
        "scheme": "NVFP4A16",
        "protection_profile": "v-first-dataflow",
    },
    "w8a16-low-rank-critical-high": {
        "scheme": "W8A16",
        "protection_profile": "low-rank-w8-critical-high",
    },
}
_CANDIDATE_SCHEMES = {
    candidate: spec["scheme"] for candidate, spec in _CANDIDATE_SPECS.items()
}
_FRESH_RELOAD_GENERATE_SEED = 20260801
_FRESH_RELOAD_PROMPT_IDS = [1, 2, 3, 4]
_FRESH_RELOAD_NEW_TOKENS = 4
_FRESH_RELOAD_WARMUP_RUNS = 1
_FRESH_RELOAD_TIMED_RUNS = 3
_RWKV7_METADATA_KEY = "rwkv7_quantization_metadata"
_LLM_COMPRESSOR_UPSTREAM_REPOSITORY = (
    "https://github.com/vllm-project/llm-compressor.git"
)
_LLM_COMPRESSOR_UPSTREAM_OID = "28c9c76b74cdd47076f95d012227482d22a8f365"
_LLM_COMPRESSOR_FORK_REPOSITORY = (
    "https://github.com/rwkv-rs/llm-compressor-rwkv.git"
)
_G1H_1_5B_CHECKPOINT = {
    "model_id": "g1h-1.5b",
    "repository": "BlinkDL/rwkv7-g1",
    "revision": "6d5762253b343eec6cfbf5ed62f872f30a4cd89c",
    "filename": "rwkv7-g1h-1.5b-20260710-ctx10240.pth",
    "sha256": "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c",
    "size_bytes": 3055444605,
}


class RWKV7RepositoryContract(BaseModel):
    """Immutable source/fork identity for this RWKV-7 adaptation."""

    model_config = ConfigDict(extra="forbid")

    upstream_repository: Literal[
        "https://github.com/vllm-project/llm-compressor.git"
    ] = _LLM_COMPRESSOR_UPSTREAM_REPOSITORY
    upstream_oid: Literal[
        "28c9c76b74cdd47076f95d012227482d22a8f365"
    ] = _LLM_COMPRESSOR_UPSTREAM_OID
    fork_repository: Literal[
        "https://github.com/rwkv-rs/llm-compressor-rwkv.git"
    ] = _LLM_COMPRESSOR_FORK_REPOSITORY


class RWKV7CheckpointContract(BaseModel):
    """Pinned real 1.5B source checkpoint and standard conversion contract."""

    model_config = ConfigDict(extra="forbid")

    model_id: Literal["g1h-1.5b"] = _G1H_1_5B_CHECKPOINT["model_id"]
    repository: Literal["BlinkDL/rwkv7-g1"] = _G1H_1_5B_CHECKPOINT["repository"]
    revision: Literal[
        "6d5762253b343eec6cfbf5ed62f872f30a4cd89c"
    ] = _G1H_1_5B_CHECKPOINT["revision"]
    filename: Literal[
        "rwkv7-g1h-1.5b-20260710-ctx10240.pth"
    ] = _G1H_1_5B_CHECKPOINT["filename"]
    sha256: Literal[
        "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c"
    ] = _G1H_1_5B_CHECKPOINT["sha256"]
    size_bytes: Literal[3055444605] = _G1H_1_5B_CHECKPOINT["size_bytes"]
    source_format: Literal["legacy_pth"] = "legacy_pth"
    converted_format: Literal["standard_hf_safetensors"] = (
        "standard_hf_safetensors"
    )
    architecture: Literal["Rwkv7ForCausalLM"] = "Rwkv7ForCausalLM"
    model_type: Literal["rwkv7"] = "rwkv7"
    embedding_layer_norm_fused: Literal[False] = False


class RWKV7LowRankW8Metadata(BaseModel):
    """Intermediate raw-Parameter packing contract, not a public loader format."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["rwkv7-low-rank-int8-groupwise-v1"] = (
        "rwkv7-low-rank-int8-groupwise-v1"
    )
    group_size: Literal[128] = _LOW_RANK_W8_GROUP_SIZE
    packed_dtype: Literal["int8"] = "int8"
    scale_dtype: Literal["float32"] = "float32"
    sidecar_filename: Literal["rwkv7_low_rank_w8.safetensors"] = (
        _LOW_RANK_W8_SIDECAR
    )
    parameter_names: list[str]
    protected_v_first_parameter_names: list[str]
    artifact_role: Literal["intermediate-serialization-only"] = (
        "intermediate-serialization-only"
    )
    standard_transformers_load_supported: Literal[False] = False
    standard_vllm_load_supported: Literal[False] = False
    required_parameter_ownership: Literal[
        "standard-quantizable-module-weight"
    ] = "standard-quantizable-module-weight"
    integration_blocker: Literal[
        "transformers-rwkv-and-vllm-module-ownership"
    ] = (
        "transformers-rwkv-and-vllm-module-ownership"
    )

    @model_validator(mode="after")
    def validate_parameter_inventory(self) -> RWKV7LowRankW8Metadata:
        if not self.parameter_names or len(self.parameter_names) != len(
            set(self.parameter_names)
        ):
            raise ValueError(
                "RWKV-7 low-rank W8 Parameters must be non-empty and unique"
            )
        low_rank_suffixes = tuple(
            f".{name}" for name in _LOW_RANK_W8_PARAMETER_NAMES
        )
        if any(not name.endswith(low_rank_suffixes) for name in self.parameter_names):
            raise ValueError("RWKV-7 low-rank W8 inventory contains a non-w/a/g tensor")
        if set(self.parameter_names) & set(self.protected_v_first_parameter_names):
            raise ValueError("RWKV-7 low-rank W8 inventory overlaps the v_first path")
        return self


class RWKV7VLLMLoaderMetadata(BaseModel):
    """Metadata consumed by the standard-HF vLLM-RWKV loader boundary."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["Rwkv7ForCausalLM"] = "Rwkv7ForCausalLM"
    model_type: Literal["rwkv7"] = "rwkv7"
    source_format: Literal["standard_hf"] = "standard_hf"
    load_format: Literal["safetensors"] = "safetensors"
    quant_method: Literal["compressed-tensors"] = "compressed-tensors"
    quantization_format: Literal["nvfp4-pack-quantized", "pack-quantized"]
    embedding_name: Literal["model.embeddings.weight"] = (
        "model.embeddings.weight"
    )
    block_prefix: Literal["model.blocks."] = "model.blocks."
    output_norm_prefix: Literal["model.ln_out."] = "model.ln_out."
    head_name: Literal["head.weight"] = "head.weight"
    legacy_pth_direct_load: Literal[False] = False
    quantized_modules: list[str]
    protected_modules: list[str]
    protected_tensors: list[str]
    low_rank_w8: RWKV7LowRankW8Metadata | None = None


class RWKV7ArtifactContract(BaseModel):
    """Self-contained loader and protection contract serialized with a result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    repository: RWKV7RepositoryContract
    checkpoint: RWKV7CheckpointContract | None
    candidate: Literal[
        "nvfp4-w4a4",
        "nvfp4-w4a16",
        "nvfp4-w4a16-protection-ablation",
        "w8a16-low-rank-critical-high",
    ]
    target_policy: QuantizationTargetPolicyMetadata
    vllm: RWKV7VLLMLoaderMetadata
    formal_checkpoint: bool
    formal_evaluation: Literal[False] = False


class RWKV7QuantizationRecipeMetadata(BaseModel):
    """Loader-facing contract for one closed RWKV-7 quantization candidate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    candidate: Literal[
        "nvfp4-w4a4",
        "nvfp4-w4a16",
        "nvfp4-w4a16-protection-ablation",
        "w8a16-low-rank-critical-high",
    ]
    candidate_order: list[str]
    algorithm: Literal["NVFP4", "INT8"]
    weight_dtype: Literal["float4", "int8"]
    weight_group_size: Literal[16, 128]
    weight_scale_dtype: Literal["float8_e4m3fn", "float32"]
    input_dtype: Literal["float4", "float16"]
    input_scale: Literal["dynamic_local", "none"]
    input_scale_dtype: Literal["float8_e4m3fn"] | None
    protection_profile: Literal[
        "critical-high",
        "v-first-dataflow",
        "low-rank-w8-critical-high",
    ]
    low_rank_weight_dtype: Literal["none", "int8"]
    targets: list[str]
    framework_versions: dict[str, str]
    quantization_applied: Literal[False] = False

    @model_validator(mode="after")
    def validate_closed_contract(self):
        if self.candidate_order != list(_CANDIDATE_SCHEMES):
            raise ValueError(
                "RWKV-7 quantization candidate set or order is unsupported"
            )
        _validate_framework_versions(self.framework_versions)
        if not self.targets:
            raise ValueError(
                "RWKV-7 quantization recipe requires resolved ChannelMix targets"
            )
        expects_low_rank_w8 = self.candidate == "w8a16-low-rank-critical-high"
        if (self.low_rank_weight_dtype == "int8") != expects_low_rank_w8:
            raise ValueError("RWKV-7 low-rank W8 recipe metadata is inconsistent")
        return self


class QuantizationTargetPolicyDecision(BaseModel):
    """One serializable selection or protection decision."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["module", "tensor"]
    names: list[str]
    reason: str


class QuantizationTargetPolicyMetadata(BaseModel):
    """Resolved evidence recorded in a serialized quantization recipe."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["rwkv7"] = "rwkv7"
    policy_version: Literal[1] = 1
    model_type: Literal["rwkv7"] = "rwkv7"
    protection_profile: Literal[
        "critical-high",
        "v-first-dataflow",
        "low-rank-w8-critical-high",
    ] = "critical-high"
    base_model_prefix: str
    selection: QuantizationTargetPolicyDecision
    parameter_selection: QuantizationTargetPolicyDecision | None = None
    protections: list[QuantizationTargetPolicyDecision]
    recipe: RWKV7QuantizationRecipeMetadata | None = None


def build_rwkv7_artifact_contract(
    target_policy: QuantizationTargetPolicyMetadata,
    candidate: str,
    *,
    checkpoint: RWKV7CheckpointContract | None = None,
) -> RWKV7ArtifactContract:
    """Resolve the exact standard-HF names protected at the runtime boundary."""

    if candidate not in _CANDIDATE_SCHEMES:
        raise ValueError(f"unsupported RWKV-7 candidate for artifact: {candidate}")
    if target_policy.recipe is None or target_policy.recipe.candidate != candidate:
        raise ValueError(
            "RWKV-7 artifact candidate must match resolved recipe metadata"
        )
    protected_modules = [
        name
        for decision in target_policy.protections
        if decision.kind == "module"
        for name in decision.names
    ]
    protected_tensors = [
        name
        for decision in target_policy.protections
        if decision.kind == "tensor"
        for name in decision.names
    ]
    parameter_selection = target_policy.parameter_selection
    if candidate == "w8a16-low-rank-critical-high":
        if parameter_selection is None or parameter_selection.kind != "tensor":
            raise ValueError("RWKV-7 low-rank W8 candidate lacks raw Parameters")
        low_rank_names = list(parameter_selection.names)
        if not low_rank_names or any(
            name.endswith((".v1", ".v2")) for name in low_rank_names
        ):
            raise ValueError("RWKV-7 low-rank W8 selection touches v_first gating")
        protected_v_first_names = sorted(
            name
            for name in protected_tensors
            if name.endswith((".v0", ".v1", ".v2"))
        )
        low_rank_w8 = RWKV7LowRankW8Metadata(
            parameter_names=low_rank_names,
            protected_v_first_parameter_names=protected_v_first_names,
        )
    else:
        if parameter_selection is not None:
            raise ValueError(
                "RWKV-7 non-W8 candidate unexpectedly selects raw Parameters"
            )
        low_rank_w8 = None
    layer_zero_value = f"{target_policy.base_model_prefix}.blocks.0.att.value"
    if layer_zero_value not in protected_modules:
        raise ValueError(
            "RWKV-7 artifact must protect the layer-0 v_first producer"
        )
    if any(name.startswith("rwkv7.") for name in target_policy.selection.names):
        raise ValueError("RWKV-7 artifact contains non-standard module names")
    return RWKV7ArtifactContract(
        repository=RWKV7RepositoryContract(),
        checkpoint=checkpoint,
        candidate=candidate,
        target_policy=target_policy,
        vllm=RWKV7VLLMLoaderMetadata(
            quantization_format=(
                "pack-quantized"
                if candidate == "w8a16-low-rank-critical-high"
                else "nvfp4-pack-quantized"
            ),
            quantized_modules=list(target_policy.selection.names),
            protected_modules=protected_modules,
            protected_tensors=protected_tensors,
            low_rank_w8=low_rank_w8,
        ),
        formal_checkpoint=checkpoint is not None,
    )


def _installed_framework_versions() -> dict[str, str]:
    import compressed_tensors
    import transformers

    return {
        "compressed_tensors": compressed_tensors.__version__,
        "transformers": transformers.__version__,
    }


def _validate_framework_versions(versions: dict[str, str]) -> None:
    if versions != _SUPPORTED_FRAMEWORK_VERSIONS:
        raise RuntimeError(
            "RWKV-7 candidate recipes require the validated framework versions: "
            f"expected={_SUPPORTED_FRAMEWORK_VERSIONS} actual={versions}"
        )


def _validate_candidate_scheme(
    candidate: str,
    scheme: Any,
    *,
    targets: list[str],
    framework_versions: dict[str, str],
) -> RWKV7QuantizationRecipeMetadata:
    weights = scheme.weights
    inputs = scheme.input_activations
    expected_inputs = candidate == "nvfp4-w4a4"
    is_w8 = candidate == "w8a16-low-rank-critical-high"
    if is_w8:
        valid_weights = (
            weights is not None
            and weights.num_bits == 8
            and str(weights.type) == "int"
            and str(weights.strategy) == "group"
            and weights.group_size == 128
            and weights.symmetric is True
        )
    else:
        valid_weights = (
            weights is not None
            and weights.num_bits == 4
            and str(weights.type) == "float"
            and str(weights.strategy) == "tensor_group"
            and weights.group_size == 16
            and str(weights.scale_dtype) == "torch.float8_e4m3fn"
        )
    valid_inputs = (inputs is not None) == expected_inputs
    if inputs is not None:
        valid_inputs = valid_inputs and (
            inputs.num_bits == 4
            and str(inputs.type) == "float"
            and str(inputs.strategy) == "tensor_group"
            and inputs.group_size == 16
            and str(inputs.dynamic) == "local"
            and str(inputs.scale_dtype) == "torch.float8_e4m3fn"
        )
    if not valid_weights or not valid_inputs:
        raise RuntimeError(
            "compressed-tensors preset "
            f"{_CANDIDATE_SCHEMES[candidate]} drifted from RWKV-7 contract"
        )

    return RWKV7QuantizationRecipeMetadata(
        candidate=candidate,
        candidate_order=list(_CANDIDATE_SCHEMES),
        algorithm="INT8" if is_w8 else "NVFP4",
        weight_dtype="int8" if is_w8 else "float4",
        weight_group_size=128 if is_w8 else 16,
        weight_scale_dtype="float32" if is_w8 else "float8_e4m3fn",
        input_dtype="float4" if inputs is not None else "float16",
        input_scale="dynamic_local" if inputs is not None else "none",
        input_scale_dtype="float8_e4m3fn" if inputs is not None else None,
        protection_profile=_CANDIDATE_SPECS[candidate]["protection_profile"],
        low_rank_weight_dtype="int8" if is_w8 else "none",
        targets=targets,
        framework_versions=framework_versions,
    )


def build_rwkv7_quantization_recipe(
    model: torch.nn.Module,
    candidate: str = "nvfp4-w4a4",
    *,
    framework_versions: dict[str, str] | None = None,
):
    """Build one validated closed-set recipe without applying quantization."""

    if candidate not in _CANDIDATE_SCHEMES:
        raise ValueError(
            f"unsupported RWKV-7 quantization candidate {candidate!r}; "
            f"allowed={list(_CANDIDATE_SCHEMES)}"
        )
    versions = (
        _installed_framework_versions()
        if framework_versions is None
        else dict(framework_versions)
    )
    _validate_framework_versions(versions)

    from llmcompressor.modifiers.quantization import QuantizationModifier

    modifier = QuantizationModifier(
        scheme=_CANDIDATE_SCHEMES[candidate],
        target_policy="rwkv7",
        target_policy_profile=_CANDIDATE_SPECS[candidate]["protection_profile"],
    )
    modifier._apply_target_policy(model)
    recipe_metadata = _validate_candidate_scheme(
        candidate,
        next(iter(modifier.resolved_config.config_groups.values())),
        targets=list(modifier.target_policy_metadata.selection.names),
        framework_versions=versions,
    )
    if (
        modifier.target_policy_metadata.protection_profile
        != recipe_metadata.protection_profile
    ):
        raise RuntimeError("RWKV-7 target protection profile drifted from candidate")
    modifier.target_policy_metadata = modifier.target_policy_metadata.model_copy(
        update={"recipe": recipe_metadata}
    )
    return modifier


def _quantize_rwkv7_low_rank_tensor(
    tensor: torch.Tensor,
    *,
    group_size: int = _LOW_RANK_W8_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim != 2 or not tensor.numel() or not torch.isfinite(tensor).all():
        raise ValueError("RWKV-7 low-rank W8 requires a finite non-empty matrix")
    if group_size < 1:
        raise ValueError("RWKV-7 low-rank W8 group_size must be positive")
    rows, columns = tensor.shape
    groups = (columns + group_size - 1) // group_size
    padded_columns = groups * group_size
    work = tensor.detach().to(device="cpu", dtype=torch.float32)
    if padded_columns != columns:
        work = torch.nn.functional.pad(work, (0, padded_columns - columns))
    grouped = work.reshape(rows, groups, group_size)
    scales = grouped.abs().amax(dim=-1) / 127.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    packed = torch.round(grouped / scales.unsqueeze(-1)).clamp(-127, 127)
    packed = packed.to(torch.int8).reshape(rows, padded_columns)[:, :columns]
    return packed.contiguous(), scales.contiguous()


def _dequantize_rwkv7_low_rank_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = _LOW_RANK_W8_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if packed.ndim != 2 or packed.dtype != torch.int8:
        raise ValueError("RWKV-7 low-rank packed tensor must be a 2D int8 matrix")
    rows, columns = packed.shape
    expected_groups = (columns + group_size - 1) // group_size
    if scales.shape != (rows, expected_groups) or scales.dtype != torch.float32:
        raise ValueError("RWKV-7 low-rank scale tensor has an invalid shape or dtype")
    expanded = scales.unsqueeze(-1).expand(rows, expected_groups, group_size)
    expanded = expanded.reshape(rows, expected_groups * group_size)[:, :columns]
    return (packed.to(torch.float32) * expanded).to(dtype=dtype)


def _apply_rwkv7_low_rank_w8(
    model: torch.nn.Module,
    metadata: RWKV7LowRankW8Metadata,
) -> dict[str, Any]:
    errors = {}
    with torch.inference_mode():
        for name in metadata.parameter_names:
            parameter = model.get_parameter(name)
            original = parameter.detach().to(torch.float32).cpu().clone()
            packed, scales = _quantize_rwkv7_low_rank_tensor(
                original,
                group_size=metadata.group_size,
            )
            dequantized = _dequantize_rwkv7_low_rank_tensor(
                packed,
                scales,
                group_size=metadata.group_size,
                dtype=parameter.dtype,
            ).to(device=parameter.device)
            parameter.copy_(dequantized)
            errors[name] = float(
                (original.cpu() - dequantized.float().cpu()).abs().max().item()
            )
    return {
        "applied": True,
        "parameter_count": len(metadata.parameter_names),
        "max_abs_error": max(errors.values(), default=0.0),
        "parameter_max_abs_error": errors,
    }


def _model_safetensor_shards(directory: Path) -> list[Path]:
    shards = [
        path
        for path in sorted(directory.glob("model*.safetensors"))
        if path.name != _LOW_RANK_W8_SIDECAR
    ]
    if not shards:
        raise RuntimeError("RWKV-7 artifact has no model safetensors shards")
    return shards


def _serialize_rwkv7_low_rank_w8(
    directory: Path,
    metadata: RWKV7LowRankW8Metadata,
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    shards = _model_safetensor_shards(directory)
    locations = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in metadata.parameter_names:
                if name in handle.keys():
                    if name in locations:
                        raise RuntimeError(
                            f"RWKV-7 low-rank Parameter is duplicated: {name}"
                        )
                    locations[name] = shard
    missing = sorted(set(metadata.parameter_names) - locations.keys())
    if missing:
        raise RuntimeError(f"RWKV-7 low-rank Parameters are missing: {missing}")

    sidecar = {}
    standard_total_size = 0
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            shard_metadata = handle.metadata()
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        for name in metadata.parameter_names:
            if locations[name] != shard:
                continue
            tensor = tensors.pop(name)
            packed, scales = _quantize_rwkv7_low_rank_tensor(
                tensor,
                group_size=metadata.group_size,
            )
            sidecar[f"{name}.packed"] = packed
            sidecar[f"{name}.scale"] = scales
        standard_total_size += sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        temporary = shard.with_name(f".{shard.name}.rwkv7.tmp")
        temporary.unlink(missing_ok=True)
        save_file(tensors, temporary, metadata=shard_metadata or {"format": "pt"})
        os.replace(temporary, shard)

    sidecar_path = directory / metadata.sidecar_filename
    temporary_sidecar = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
    temporary_sidecar.unlink(missing_ok=True)
    save_file(sidecar, temporary_sidecar, metadata={"format": "pt"})
    os.replace(temporary_sidecar, sidecar_path)

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json_file(index_path, "RWKV-7 safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError("RWKV-7 safetensors index has no weight_map")
        for name in metadata.parameter_names:
            if weight_map.pop(name, None) is None:
                raise RuntimeError(
                    f"RWKV-7 safetensors index lacks low-rank Parameter: {name}"
                )
        index_metadata = index.setdefault("metadata", {})
        if isinstance(index_metadata, dict):
            index_metadata["total_size"] = standard_total_size
        _atomic_json(index_path, index)

    packed_size = sum(
        tensor.numel()
        for tensor in sidecar.values()
        if tensor.dtype == torch.int8
    )
    scale_size = sum(
        tensor.numel() * tensor.element_size()
        for tensor in sidecar.values()
        if tensor.dtype == torch.float32
    )
    return {
        "sidecar_filename": metadata.sidecar_filename,
        "sidecar_sha256": _sha256_file(sidecar_path),
        "parameter_count": len(metadata.parameter_names),
        "packed_bytes": packed_size,
        "scale_bytes": scale_size,
    }


def _iter_rwkv7_low_rank_w8_dequantized(
    directory: Path | str,
    metadata: RWKV7LowRankW8Metadata | dict[str, Any],
    *,
    dtype: torch.dtype = torch.bfloat16,
):
    """Inspect an intermediate sidecar without defining a model loading path."""
    from safetensors.torch import load_file

    directory = Path(directory)
    contract = RWKV7LowRankW8Metadata.model_validate(metadata)
    sidecar = load_file(directory / contract.sidecar_filename, device="cpu")
    expected_keys = {
        f"{name}.{suffix}"
        for name in contract.parameter_names
        for suffix in ("packed", "scale")
    }
    if set(sidecar) != expected_keys:
        raise RuntimeError("RWKV-7 low-rank W8 sidecar tensor inventory drifted")
    for name in contract.parameter_names:
        yield name, _dequantize_rwkv7_low_rank_tensor(
            sidecar[f"{name}.packed"],
            sidecar[f"{name}.scale"],
            group_size=contract.group_size,
            dtype=dtype,
        )


def validate_rwkv7_low_rank_standard_load(
    loading_info: dict[str, Any],
    metadata: RWKV7LowRankW8Metadata | dict[str, Any],
) -> dict[str, Any]:
    """Fail closed until the public Transformers loader owns low-rank W8."""

    contract = RWKV7LowRankW8Metadata.model_validate(metadata)
    missing_keys = loading_info.get("missing_keys")
    if not isinstance(missing_keys, (list, tuple, set, frozenset)) or not all(
        isinstance(name, str) for name in missing_keys
    ):
        raise RuntimeError(
            "standard Transformers loading info lacks a valid missing_keys inventory"
        )
    missing_low_rank = sorted(set(contract.parameter_names) & set(missing_keys))
    if missing_low_rank:
        raise RuntimeError(
            "standard Transformers loader did not restore RWKV-7 low-rank W8 "
            f"Parameters: {missing_low_rank}"
        )
    if not contract.standard_transformers_load_supported:
        raise RuntimeError(
            "RWKV-7 low-rank W8 is an intermediate serialization artifact; "
            "standard Transformers module/weight ownership is not integrated"
        )
    return {
        "passed": True,
        "format": contract.format,
        "parameter_count": len(contract.parameter_names),
        "parameter_names": contract.parameter_names,
        "standard_transformers_load_supported": True,
    }


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def _audit_rwkv7_low_rank_w8(
    directory: Path,
    metadata: RWKV7LowRankW8Metadata,
) -> dict[str, Any]:
    from safetensors import safe_open

    model_tensor_names = set()
    for shard in _model_safetensor_shards(directory):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            model_tensor_names.update(handle.keys())
    retained = sorted(set(metadata.parameter_names) & model_tensor_names)
    if retained:
        raise RuntimeError(
            f"RWKV-7 low-rank raw Parameters remained uncompressed: {retained}"
        )
    restored = list(
        _iter_rwkv7_low_rank_w8_dequantized(
            directory,
            metadata,
            dtype=torch.float32,
        )
    )
    return {
        "format": metadata.format,
        "parameter_count": len(restored),
        "parameter_names": [name for name, _ in restored],
        "all_finite": all(torch.isfinite(tensor).all() for _, tensor in restored),
        "v_first_protected": not (
            set(metadata.parameter_names)
            & set(metadata.protected_v_first_parameter_names)
        ),
        "artifact_role": metadata.artifact_role,
        "standard_transformers_load_supported": (
            metadata.standard_transformers_load_supported
        ),
        "standard_vllm_load_supported": metadata.standard_vllm_load_supported,
    }


def audit_rwkv7_quantized_checkpoint(
    output_dir: Path,
    expected_targets: list[str],
    candidate: str,
    artifact_contract: RWKV7ArtifactContract | None = None,
) -> dict[str, Any]:
    """Verify compressed tensor storage, not merely serialized recipe metadata."""
    from safetensors import safe_open

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config", {})
    serialized_contract = config.get(_RWKV7_METADATA_KEY)
    if artifact_contract is not None:
        loaded_contract = RWKV7ArtifactContract.model_validate(serialized_contract)
        if loaded_contract != artifact_contract:
            raise RuntimeError("RWKV-7 serialized artifact contract drifted")
    expected_format = (
        "pack-quantized"
        if candidate == "w8a16-low-rank-critical-high"
        else "nvfp4-pack-quantized"
    )
    if (
        quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("quantization_status") != "compressed"
        or quantization.get("format") != expected_format
    ):
        raise RuntimeError("RWKV-7 checkpoint lacks expected compression metadata")
    groups = quantization.get("config_groups", {})
    if not isinstance(groups, dict) or len(groups) != 1:
        raise RuntimeError("RWKV-7 checkpoint has an invalid quantization config group")
    group = next(iter(groups.values()))
    input_quantized = group.get("input_activations") is not None
    if input_quantized != (candidate == "nvfp4-w4a4"):
        raise RuntimeError(
            "RWKV-7 checkpoint activation quantization differs from candidate"
        )
    tensors: dict[str, tuple[list[int], str]] = {}
    for shard in sorted(output_dir.glob("*.safetensors")):
        if shard.name == _LOW_RANK_W8_SIDECAR:
            continue
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensors[name] = (
                    handle.get_slice(name).get_shape(),
                    handle.get_slice(name).get_dtype(),
                )
    for target in expected_targets:
        if candidate == "w8a16-low-rank-critical-high":
            required = {
                f"{target}.weight_packed",
                f"{target}.weight_scale",
                f"{target}.weight_shape",
            }
        else:
            required = {
                f"{target}.weight_packed",
                f"{target}.weight_scale",
                f"{target}.weight_global_scale",
            }
            if candidate == "nvfp4-w4a4":
                required.add(f"{target}.input_global_scale")
        if not input_quantized and f"{target}.input_global_scale" in tensors:
            raise RuntimeError(f"weight-only target quantized its input: {target}")
        missing = sorted(required - tensors.keys())
        if missing or f"{target}.weight" in tensors:
            raise RuntimeError(
                "RWKV-7 target was not physically compressed: "
                f"{target}; missing={missing}"
            )
        expected_packed_dtype = (
            "I32" if candidate == "w8a16-low-rank-critical-high" else "U8"
        )
        if tensors[f"{target}.weight_packed"][1] != expected_packed_dtype:
            raise RuntimeError(
                f"RWKV-7 target has drifted packed dtype: {target}"
            )
        if (
            candidate != "w8a16-low-rank-critical-high"
            and tensors[f"{target}.weight_scale"][1] != "F8_E4M3"
        ):
            raise RuntimeError(f"RWKV-7 target has drifted scale dtype: {target}")
    protected_names = set()
    if artifact_contract is not None:
        protected_names.update(artifact_contract.vllm.protected_tensors)
        protected_names.update(
            f"{name}.weight" for name in artifact_contract.vllm.protected_modules
        )
        for name in artifact_contract.vllm.protected_modules:
            if any(key.startswith(f"{name}.weight_") for key in tensors):
                raise RuntimeError(
                    f"RWKV-7 protected module was compressed: {name}"
                )
    protected = sorted(protected_names & tensors.keys())
    low_rank_w8 = None
    if artifact_contract is not None and artifact_contract.vllm.low_rank_w8:
        low_rank_w8 = _audit_rwkv7_low_rank_w8(
            output_dir,
            artifact_contract.vllm.low_rank_w8,
        )
    return {
        "format": expected_format,
        "targets": expected_targets,
        "protected_tensor_count": len(protected),
        "tensor_count": len(tensors),
        "input_quantized": input_quantized,
        "artifact_contract_serialized": artifact_contract is not None,
        "low_rank_w8": low_rank_w8,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _fresh_reload_generate_script() -> str:
    """Return the isolated loader/generation program used for formal evidence."""

    return r"""
import json, math, statistics, sys, time, torch
from llmcompressor.modifiers.quantization.rwkv7 import (
    validate_rwkv7_low_rank_standard_load,
)
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

config = AutoConfig.from_pretrained(sys.argv[1])
generate_seed = int(sys.argv[2])
prompt_ids = json.loads(sys.argv[3])
max_new_tokens = int(sys.argv[4])
warmup_runs = int(sys.argv[5])
timed_runs = int(sys.argv[6])
runtime_dtype = config.dtype
assert isinstance(runtime_dtype, torch.dtype)
assert warmup_runs >= 1
assert timed_runs >= 1
contract = getattr(config, 'rwkv7_quantization_metadata')
assert contract['schema_version'] == 1
assert contract['candidate'] in (
    'nvfp4-w4a4',
    'nvfp4-w4a16',
    'nvfp4-w4a16-protection-ablation',
    'w8a16-low-rank-critical-high',
)
assert contract['vllm']['architecture'] == 'Rwkv7ForCausalLM'
assert contract['vllm']['source_format'] == 'standard_hf'
assert contract['vllm']['legacy_pth_direct_load'] is False

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
load_started = time.perf_counter()
model, loading_info = AutoModelForCausalLM.from_pretrained(
    sys.argv[1],
    device_map='cuda',
    dtype=runtime_dtype,
    quantization_config=CompressedTensorsConfig(dequantize=True),
    output_loading_info=True,
)
low_rank_contract = contract['vllm'].get('low_rank_w8')
low_rank_standard_load = (
    validate_rwkv7_low_rank_standard_load(loading_info, low_rank_contract)
    if low_rank_contract is not None
    else None
)
model = model.to(dtype=runtime_dtype).eval()
torch.cuda.synchronize()
load_latency_ms = (time.perf_counter() - load_started) * 1000.0
load_peak_allocated_bytes = torch.cuda.max_memory_allocated()
load_peak_reserved_bytes = torch.cuda.max_memory_reserved()
model_resident_allocated_bytes = torch.cuda.memory_allocated()
model_resident_reserved_bytes = torch.cuda.memory_reserved()

quantized = [
    model.get_submodule(name) for name in contract['vllm']['quantized_modules']
]
protected = [
    model.get_submodule(name) for name in contract['vllm']['protected_modules']
]
protected_tensors = [
    model.get_parameter(name) for name in contract['vllm']['protected_tensors']
]
assert all(
    getattr(module, 'quantization_scheme', None) is not None for module in quantized
)
assert all(module.weight.dtype == runtime_dtype for module in quantized)
assert all(getattr(module, 'quantization_scheme', None) is None for module in protected)
assert all(module.weight.dtype == runtime_dtype for module in protected)
assert all(parameter.dtype == runtime_dtype for parameter in protected_tensors)

prompt = torch.tensor([prompt_ids], device='cuda')

def generate_once():
    torch.manual_seed(generate_seed)
    return model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_k=8,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=[],
    )

with torch.inference_mode():
    logits = model(prompt).logits
    for _ in range(warmup_runs):
        generate_once()
    torch.cuda.synchronize()
    generate_baseline_allocated_bytes = torch.cuda.memory_allocated()
    generate_baseline_reserved_bytes = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    latencies_ms = []
    generated = None
    for _ in range(timed_runs):
        torch.cuda.synchronize()
        generate_started = time.perf_counter()
        generated = generate_once()
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - generate_started) * 1000.0)

assert generated is not None
assert torch.isfinite(logits).all()
assert generated.shape == (1, len(prompt_ids) + max_new_tokens)
assert generated[0, :len(prompt_ids)].tolist() == prompt_ids
sorted_latencies_ms = sorted(latencies_ms)
latency_p90_index = max(0, math.ceil(0.9 * timed_runs) - 1)
elapsed_seconds = sum(latencies_ms) / 1000.0
print(json.dumps({
    'dtype': str(runtime_dtype),
    'logits_dtype': str(logits.dtype),
    'quantized_module_count': len(quantized),
    'protected_module_count': len(protected),
    'protected_tensor_count': len(protected_tensors),
    'artifact_contract_validated': True,
    'low_rank_w8': low_rank_standard_load,
    'runtime_measurement': {
        'scope': 'fresh-process-transformers-generate-diagnostic',
        'canonical_performance_acceptance': False,
        'device_name': torch.cuda.get_device_name(),
        'device_capability': list(torch.cuda.get_device_capability()),
        'load_latency_ms': load_latency_ms,
        'load_peak_allocated_bytes': load_peak_allocated_bytes,
        'load_peak_reserved_bytes': load_peak_reserved_bytes,
        'model_resident_allocated_bytes': model_resident_allocated_bytes,
        'model_resident_reserved_bytes': model_resident_reserved_bytes,
        'generate': {
            'warmup_runs': warmup_runs,
            'timed_runs': timed_runs,
            'new_tokens_per_run': max_new_tokens,
            'latencies_ms': latencies_ms,
            'latency_p50_ms': statistics.median(latencies_ms),
            'latency_p90_ms': sorted_latencies_ms[latency_p90_index],
            'throughput_tokens_per_second': (
                timed_runs * max_new_tokens / elapsed_seconds
            ),
            'baseline_allocated_bytes': generate_baseline_allocated_bytes,
            'baseline_reserved_bytes': generate_baseline_reserved_bytes,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        },
    },
    'standard_generate': {
        'passed': True,
        'use_cache': True,
        'seed': generate_seed,
        'prompt_ids': prompt_ids,
        'max_new_tokens': max_new_tokens,
        'generated_ids': generated.tolist(),
    },
}))
"""


def quantize_rwkv7_oneshot(
    model_factory: Callable[[], torch.nn.Module],
    output_dir: Path,
    *,
    calibration_dataset: object | None,
    processor: object | None,
    candidates: tuple[str, ...] = tuple(_CANDIDATE_SCHEMES),
    forced_candidate: str | None = None,
    checkpoint_contract: RWKV7CheckpointContract | None = None,
    fresh_reload_prompt_ids: list[int] | None = None,
    fresh_reload_new_tokens: int = _FRESH_RELOAD_NEW_TOKENS,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Execute the closed candidate order through standard ``oneshot``."""
    from llmcompressor import oneshot

    if candidates != tuple(_CANDIDATE_SCHEMES):
        raise ValueError(
            "RWKV-7 quantization execution requires the closed candidate order"
        )
    if forced_candidate is not None and forced_candidate not in _CANDIDATE_SCHEMES:
        raise ValueError(f"unsupported forced RWKV-7 candidate: {forced_candidate}")
    prompt_ids = list(
        _FRESH_RELOAD_PROMPT_IDS
        if fresh_reload_prompt_ids is None
        else fresh_reload_prompt_ids
    )
    if not prompt_ids or any(
        not isinstance(token, int) or token < 0 for token in prompt_ids
    ):
        raise ValueError("fresh reload prompt IDs must be non-empty non-negative ints")
    if fresh_reload_new_tokens < 1:
        raise ValueError("fresh reload must generate at least one token")
    execution_candidates = (
        candidates if forced_candidate is None else (forced_candidate,)
    )
    failures = []
    for candidate in execution_candidates:
        model = model_factory()
        modifier = build_rwkv7_quantization_recipe(model, candidate)
        artifact_contract = build_rwkv7_artifact_contract(
            modifier.target_policy_metadata,
            candidate,
            checkpoint=checkpoint_contract,
        )
        destination = output_dir / candidate
        destination.mkdir(parents=True, exist_ok=True)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            quantization_started = time.perf_counter()
            result = oneshot(
                model=model,
                dataset=calibration_dataset if candidate == "nvfp4-w4a4" else None,
                processor=processor if candidate == "nvfp4-w4a4" else None,
                recipe=modifier,
                pipeline="basic" if candidate == "nvfp4-w4a4" else "datafree",
                output_dir=None,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            quantization_runtime = {
                "scope": "llmcompressor-oneshot",
                "latency_ms": (
                    time.perf_counter() - quantization_started
                ) * 1000.0,
                "cuda_measured": torch.cuda.is_available(),
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else None
                ),
                "peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved()
                    if torch.cuda.is_available()
                    else None
                ),
            }
            low_rank_runtime = None
            if artifact_contract.vllm.low_rank_w8 is not None:
                low_rank_runtime = _apply_rwkv7_low_rank_w8(
                    result,
                    artifact_contract.vllm.low_rank_w8,
                )
            base_model = result.base_model
            first_channel_mix = base_model.blocks[0].ffn
            reference = next(result.parameters())
            with torch.inference_mode():
                cell_output, cell_state = first_channel_mix(
                    torch.randn(
                        1,
                        4,
                        result.config.hidden_size,
                        device=reference.device,
                        dtype=reference.dtype,
                    ),
                    torch.zeros(
                        1,
                        result.config.hidden_size,
                        device=reference.device,
                        dtype=reference.dtype,
                    ),
                )
            if (
                not torch.isfinite(cell_output).all()
                or not torch.isfinite(cell_state).all()
            ):
                raise RuntimeError(
                    "quantized RWKV-7 ChannelMix cell produced non-finite output"
                )
            setattr(
                result.config,
                _RWKV7_METADATA_KEY,
                artifact_contract.model_dump(mode="json"),
            )
            result.save_pretrained(destination, save_compressed=True)
            low_rank_serialization = None
            if artifact_contract.vllm.low_rank_w8 is not None:
                low_rank_serialization = _serialize_rwkv7_low_rank_w8(
                    destination,
                    artifact_contract.vllm.low_rank_w8,
                )
            if processor is not None and hasattr(processor, "save_pretrained"):
                processor.save_pretrained(destination)
            audit = audit_rwkv7_quantized_checkpoint(
                destination,
                modifier.target_policy_metadata.selection.names,
                candidate,
                artifact_contract,
            )
        except Exception as error:
            failures.append(
                {
                    "candidate": candidate,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        reload_environment = dict(os.environ)
        reload_temporary = destination / ".fresh-reload-tmp"
        reload_temporary.mkdir(exist_ok=True)
        reload_environment["TMPDIR"] = str(reload_temporary)
        reload_run = subprocess.run(
            [
                sys.executable,
                "-c",
                _fresh_reload_generate_script(),
                str(destination),
                str(_FRESH_RELOAD_GENERATE_SEED),
                json.dumps(prompt_ids),
                str(fresh_reload_new_tokens),
                str(_FRESH_RELOAD_WARMUP_RUNS),
                str(_FRESH_RELOAD_TIMED_RUNS),
            ],
            capture_output=True,
            text=True,
            env=reload_environment,
        )
        reload_evidence = None
        if reload_run.returncode == 0:
            reload_evidence = json.loads(reload_run.stdout.strip().splitlines()[-1])
        metadata = {
            "schema_version": 1,
            "candidate": candidate,
            "candidate_order": list(candidates),
            "forced_candidate": forced_candidate,
            "quantization_applied": True,
            "artifact_contract": artifact_contract.model_dump(mode="json"),
            "audit": audit,
            "cell_forward": {
                "passed": True,
                "output_shape": list(cell_output.shape),
                "state_shape": list(cell_state.shape),
            },
            "quantization_runtime": quantization_runtime,
            "low_rank_w8": {
                "runtime": low_rank_runtime,
                "serialization": low_rank_serialization,
            },
            "fresh_reload": {
                "passed": reload_run.returncode == 0,
                "returncode": reload_run.returncode,
                "stderr": reload_run.stderr[-8000:],
                "evidence": reload_evidence,
                "source_owner": "Transformers RWKV7 loader",
                "regression_expectation": (
                    "the standard compressed-tensors dequantization path must restore "
                    "packed Linear weights at the checkpoint dtype before forward, "
                    "and standard generate(use_cache=True) must propagate recurrent "
                    "state through decode"
                ),
            },
            "failures": failures,
        }
        _atomic_json(destination / "rwkv7_quantization_execution.json", metadata)
        if checkpoint_contract is not None and reload_run.returncode != 0:
            failures.append(
                {
                    "candidate": candidate,
                    "stage": "fresh_reload_generate",
                    "error_type": "SubprocessError",
                    "error": reload_run.stderr[-8000:],
                }
            )
            _atomic_json(
                destination / "rwkv7_quantization_execution.json", metadata
            )
            continue
        return result, metadata
    raise RuntimeError(f"all RWKV-7 quantization candidates failed: {failures}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_rwkv7_checkpoint(checkpoint_path: Path) -> RWKV7CheckpointContract:
    """Fail closed unless ``checkpoint_path`` is the pinned real g1h 1.5B file."""

    checkpoint_path = checkpoint_path.resolve()
    contract = RWKV7CheckpointContract()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RWKV-7 checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.name != contract.filename:
        raise ValueError(
            "RWKV-7 formal candidate runner only accepts the pinned checkpoint: "
            f"expected={contract.filename} actual={checkpoint_path.name}"
        )
    actual_size = checkpoint_path.stat().st_size
    if actual_size != contract.size_bytes:
        raise ValueError(
            "RWKV-7 checkpoint size mismatch: "
            f"expected={contract.size_bytes} actual={actual_size}"
        )
    actual_sha256 = _sha256_file(checkpoint_path)
    if actual_sha256 != contract.sha256:
        raise ValueError(
            "RWKV-7 checkpoint SHA-256 mismatch: "
            f"expected={contract.sha256} actual={actual_sha256}"
        )
    return contract


def _artifact_file_manifest(directory: Path) -> dict[str, Any]:
    files = []
    for path in sorted(
        candidate for candidate in directory.rglob("*") if candidate.is_file()
    ):
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"artifact directory is empty: {directory}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": files,
        "file_count": len(files),
        "size_bytes": sum(item["size_bytes"] for item in files),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _load_calibration_records(
    calibration_path: Path,
    *,
    expected_sha256: str,
    max_samples: int,
    max_length: int,
    vocab_size: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    calibration_path = calibration_path.resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"calibration JSONL does not exist: {calibration_path}")
    actual_sha256 = _sha256_file(calibration_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "calibration SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    if max_samples < 1 or max_length < 1:
        raise ValueError("calibration max_samples and max_length must be positive")

    records = []
    with calibration_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            input_ids = payload.get("input_ids")
            if (
                not isinstance(input_ids, list)
                or not input_ids
                or any(
                    not isinstance(token, int) or not 0 <= token < vocab_size
                    for token in input_ids
                )
            ):
                raise ValueError(
                    f"calibration line {line_number} has invalid input_ids"
                )
            input_ids = input_ids[:max_length]
            records.append(
                {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
                }
            )
            if len(records) == max_samples:
                break
    if not records:
        raise ValueError("calibration JSONL contains no usable records")
    return records, {
        "path": str(calibration_path),
        "sha256": actual_sha256,
        "sample_count": len(records),
        "max_samples": max_samples,
        "max_length": max_length,
        "format": "jsonl-input_ids-v1",
    }


def _prepare_standard_rwkv7_checkpoint(
    checkpoint_path: Path,
    destination: Path,
    checkpoint_contract: RWKV7CheckpointContract,
) -> dict[str, Any]:
    provenance_path = destination / "rwkv7_source_provenance.json"
    expected_provenance = checkpoint_contract.model_dump(mode="json")
    if destination.exists() and any(destination.iterdir()):
        if not provenance_path.is_file():
            raise RuntimeError(
                "standard checkpoint destination is non-empty without provenance"
            )
        actual_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if actual_provenance != expected_provenance:
            raise RuntimeError("standard checkpoint provenance does not match source")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import (
            convert_rwkv7_checkpoint_to_hf_format,
        )

        convert_rwkv7_checkpoint_to_hf_format(
            str(checkpoint_path),
            str(destination),
            dtype="bfloat16",
            safe_serialization=True,
            fuse_embedding_layer_norm=False,
        )
        _atomic_json(provenance_path, expected_provenance)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(destination)
    if (
        config.model_type != checkpoint_contract.model_type
        or config.architectures != [checkpoint_contract.architecture]
        or bool(getattr(config, "embedding_layer_norm_fused", False))
    ):
        raise RuntimeError(
            "converted checkpoint does not satisfy the standard RWKV-7 loader contract"
        )
    if not list(destination.glob("*.safetensors")):
        raise RuntimeError("converted checkpoint has no safetensors weights")
    return _artifact_file_manifest(destination)


def run_rwkv7_checkpoint_candidate(
    checkpoint_path: Path,
    calibration_path: Path,
    output_dir: Path,
    *,
    calibration_sha256: str,
    implementation_revision: str,
    candidate: Literal[
        "nvfp4-w4a4",
        "nvfp4-w4a16",
        "nvfp4-w4a16-protection-ablation",
        "w8a16-low-rank-critical-high",
    ],
    max_calibration_samples: int = 128,
    max_calibration_length: int = 1024,
) -> dict[str, Any]:
    """Quantize the pinned 1.5B checkpoint and emit a traceable candidate artifact."""

    if not re.fullmatch(r"[0-9a-f]{40}", implementation_revision):
        raise ValueError("implementation_revision must be a full lowercase Git OID")
    if candidate not in _CANDIDATE_SCHEMES:
        raise ValueError(f"unsupported RWKV-7 candidate: {candidate}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12:
        raise RuntimeError("formal RWKV-7 NVFP4 execution requires a Blackwell GPU")

    checkpoint_contract = verify_rwkv7_checkpoint(checkpoint_path)
    output_dir = output_dir.resolve()
    standard_checkpoint = output_dir / "baseline-standard-hf"
    standard_manifest = _prepare_standard_rwkv7_checkpoint(
        checkpoint_path.resolve(), standard_checkpoint, checkpoint_contract
    )

    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(standard_checkpoint)
    records, calibration = _load_calibration_records(
        calibration_path,
        expected_sha256=calibration_sha256,
        max_samples=max_calibration_samples,
        max_length=max_calibration_length,
        vocab_size=config.vocab_size,
    )
    calibration_loader = DataLoader(records, batch_size=None)

    def model_factory():
        model = AutoModelForCausalLM.from_pretrained(
            standard_checkpoint,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        if model.config.model_type != "rwkv7":
            raise RuntimeError("standard checkpoint loaded a non-RWKV-7 model")
        return model

    prompt_ids = records[0]["input_ids"][0, :16].tolist()
    _, execution = quantize_rwkv7_oneshot(
        model_factory,
        output_dir / "candidates",
        calibration_dataset=calibration_loader,
        processor=None,
        forced_candidate=candidate,
        checkpoint_contract=checkpoint_contract,
        fresh_reload_prompt_ids=prompt_ids,
    )
    candidate_dir = output_dir / "candidates" / candidate
    result = {
        "schema_version": 1,
        "implementation_revision": implementation_revision,
        "repository": RWKV7RepositoryContract().model_dump(mode="json"),
        "checkpoint": checkpoint_contract.model_dump(mode="json"),
        "standard_checkpoint": standard_manifest,
        "calibration": calibration,
        "candidate": candidate,
        "execution": execution,
        "candidate_artifact": _artifact_file_manifest(candidate_dir),
        "formal_checkpoint": True,
        "formal_evaluation": False,
        "diagnostic_tiny": False,
    }
    _atomic_json(output_dir / "rwkv7_candidate_result.json", result)
    return result


def _get_rwkv7_model_types() -> tuple[type, type[torch.nn.Module]]:
    try:
        from transformers import Rwkv7Config
        from transformers.models.rwkv7 import Rwkv7ForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "The `rwkv7` target policy requires Transformers with the standard "
            "`Rwkv7Config` and `Rwkv7ForCausalLM` APIs."
        ) from error

    return Rwkv7Config, Rwkv7ForCausalLM


def _resolve_standard_base_model(
    model: torch.nn.Module,
) -> tuple[str, torch.nn.Module]:
    config_type, causal_lm_type = _get_rwkv7_model_types()
    config = getattr(model, "config", None)
    if not isinstance(config, config_type) or config.model_type != "rwkv7":
        raise ValueError(
            "The `rwkv7` target policy only accepts a model configured by the "
            "standard Transformers `Rwkv7Config`."
        )
    if not isinstance(model, causal_lm_type):
        raise ValueError(
            "The `rwkv7` target policy only accepts the standard Transformers "
            "`Rwkv7ForCausalLM` model."
        )

    expected_prefix = getattr(causal_lm_type, "base_model_prefix", None)
    actual_prefix = getattr(model, "base_model_prefix", None)
    if (
        not isinstance(expected_prefix, str)
        or not expected_prefix
        or actual_prefix != expected_prefix
    ):
        raise ValueError(
            "RWKV-7 target policy requires `base_model_prefix` to match the "
            f"standard Rwkv7ForCausalLM declaration ({expected_prefix!r}), got "
            f"{actual_prefix!r}."
        )

    base_model = getattr(model, "base_model", None)
    registered_base_model = getattr(model, actual_prefix, None)
    if (
        not isinstance(base_model, torch.nn.Module)
        or base_model is not registered_base_model
    ):
        raise ValueError(
            "RWKV-7 target policy requires the standard `base_model` API to "
            f"resolve to the registered `{actual_prefix}` submodule."
        )
    if getattr(base_model, "config", None) is not config:
        raise ValueError(
            "RWKV-7 target policy requires the resolved base model to share the "
            "causal LM's `Rwkv7Config` instance."
        )

    return actual_prefix, base_model


def _attention_ignore(base_model_prefix: str) -> str:
    return (
        rf"re:^{re.escape(base_model_prefix)}\.blocks\.\d+\.att\."
        r"(receptance|key|value|output)$"
    )


def _attention_value_ignore(base_model_prefix: str) -> str:
    return rf"re:^{re.escape(base_model_prefix)}\.blocks\.\d+\.att\.value$"


def _require_module(
    parent: torch.nn.Module,
    name: str,
    expected_type: type[torch.nn.Module],
    path: str,
) -> torch.nn.Module:
    module = getattr(parent, name, None)
    if not isinstance(module, expected_type):
        raise ValueError(
            f"RWKV-7 target policy requires `{path}` to be {expected_type.__name__}."
        )
    return module


def _require_parameter(parent: torch.nn.Module, name: str, path: str) -> None:
    if not isinstance(getattr(parent, name, None), torch.nn.Parameter):
        raise ValueError(
            f"RWKV-7 target policy requires recurrent tensor `{path}` to be "
            "a Parameter."
        )


def _validate_policy_inputs(
    resolved_targets: set[str],
    ignore: list[str],
    kv_cache_enabled: bool,
    required_ignore: list[str],
) -> None:
    if resolved_targets != {"Linear"}:
        raise ValueError(
            "The `rwkv7` target policy owns module selection and requires the "
            "resolved quantization targets to be exactly ['Linear']."
        )
    if kv_cache_enabled:
        raise ValueError(
            "The `rwkv7` target policy does not accept `kv_cache_scheme`; RWKV-7 "
            "uses recurrent WKV state instead of a transformer KV cache."
        )

    allowed_ignore = set(required_ignore)
    unsupported_ignore = sorted(set(ignore) - allowed_ignore)
    if unsupported_ignore:
        raise ValueError(
            "The `rwkv7` target policy owns its fail-closed protection set; "
            f"unsupported ignore entries: {unsupported_ignore}."
        )


def apply_rwkv7_target_policy(
    model: torch.nn.Module,
    resolved_targets: set[str],
    ignore: list[str],
    kv_cache_enabled: bool,
    protection_profile: Literal[
        "critical-high",
        "v-first-dataflow",
        "low-rank-w8-critical-high",
    ] = "critical-high",
) -> tuple[list[str], QuantizationTargetPolicyMetadata]:
    """Validate standard RWKV-7 structure and select only ChannelMix linears.

    The validation is intentionally completed before the quantization config is
    applied. Any architecture drift therefore fails without partially modifying
    the model.
    """

    base_model_prefix, base_model = _resolve_standard_base_model(model)
    attention_ignore = _attention_ignore(base_model_prefix)
    attention_value_ignore = _attention_value_ignore(base_model_prefix)
    if protection_profile in {"critical-high", "low-rank-w8-critical-high"}:
        required_ignore = [attention_ignore, _HEAD_IGNORE]
    elif protection_profile == "v-first-dataflow":
        required_ignore = [attention_value_ignore, _HEAD_IGNORE]
    else:
        raise ValueError(
            f"unsupported RWKV-7 protection profile: {protection_profile}"
        )
    _validate_policy_inputs(
        resolved_targets,
        ignore,
        kv_cache_enabled,
        required_ignore,
    )
    config = model.config
    blocks = _require_module(
        base_model,
        "blocks",
        torch.nn.ModuleList,
        f"{base_model_prefix}.blocks",
    )
    if len(blocks) != config.num_hidden_layers:
        raise ValueError(
            f"RWKV-7 target policy requires `{base_model_prefix}.blocks` length "
            "to match "
            f"`num_hidden_layers` ({config.num_hidden_layers}), got {len(blocks)}."
        )

    selected_modules: list[str] = []
    first_value_module: list[str] = []
    later_value_modules: list[str] = []
    other_time_mix_modules: list[str] = []
    later_value_tensors: list[str] = []
    recurrent_time_mix_tensors: list[str] = []
    selected_low_rank_tensors: list[str] = []
    expected_linear_modules = {_HEAD_IGNORE}

    _require_module(model, "head", torch.nn.Linear, "head")
    for layer_id, block in enumerate(blocks):
        block_path = f"{base_model_prefix}.blocks.{layer_id}"

        attention = _require_module(block, "att", torch.nn.Module, f"{block_path}.att")
        channel_mix = _require_module(
            block, "ffn", torch.nn.Module, f"{block_path}.ffn"
        )
        if getattr(attention, "layer_id", None) != layer_id:
            raise ValueError(
                f"RWKV-7 target policy requires `{block_path}.att.layer_id == "
                f"{layer_id}`."
            )

        for parameter_name in _TIME_MIX_PARAMETER_NAMES:
            parameter_path = f"{block_path}.att.{parameter_name}"
            _require_parameter(
                attention,
                parameter_name,
                parameter_path,
            )
            if (
                protection_profile == "low-rank-w8-critical-high"
                and parameter_name in _LOW_RANK_W8_PARAMETER_NAMES
            ):
                selected_low_rank_tensors.append(parameter_path)
            else:
                recurrent_time_mix_tensors.append(parameter_path)
        for parameter_name in ("v0", "v1", "v2"):
            parameter_path = f"{block_path}.att.{parameter_name}"
            if layer_id == 0:
                if hasattr(attention, parameter_name):
                    raise ValueError(
                        "RWKV-7 target policy requires layer 0 to produce "
                        f"`v_first` without `{parameter_path}`."
                    )
            else:
                _require_parameter(attention, parameter_name, parameter_path)
                later_value_tensors.append(parameter_path)

        for linear_name in _TIME_MIX_LINEAR_NAMES:
            module_path = f"{block_path}.att.{linear_name}"
            _require_module(attention, linear_name, torch.nn.Linear, module_path)
            expected_linear_modules.add(module_path)
            if linear_name == "value" and layer_id == 0:
                first_value_module.append(module_path)
            elif linear_name == "value":
                later_value_modules.append(module_path)
            else:
                other_time_mix_modules.append(module_path)

        _require_parameter(channel_mix, "x_k", f"{block_path}.ffn.x_k")
        for linear_name in ("key", "value"):
            module_path = f"{block_path}.ffn.{linear_name}"
            _require_module(channel_mix, linear_name, torch.nn.Linear, module_path)
            selected_modules.append(module_path)
            expected_linear_modules.add(module_path)

    actual_linear_modules = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    if actual_linear_modules != expected_linear_modules:
        missing = sorted(expected_linear_modules - actual_linear_modules)
        unexpected = sorted(actual_linear_modules - expected_linear_modules)
        raise ValueError(
            "RWKV-7 target policy found non-standard Linear modules; "
            f"missing={missing}, unexpected={unexpected}."
        )

    if protection_profile == "v-first-dataflow":
        selected_modules.extend(other_time_mix_modules)

    policy_ignore = list(dict.fromkeys([*ignore, *required_ignore]))
    protections = [
        QuantizationTargetPolicyDecision(
            kind="module",
            names=first_value_module,
            reason=(
                "Layer-0 TimeMix value projection produces v_first, which is "
                "consumed by every later RWKV-7 block."
            ),
        ),
        QuantizationTargetPolicyDecision(
            kind="module",
            names=later_value_modules,
            reason=(
                "Later TimeMix value projections are blended with v_first before "
                "entering recurrent WKV state updates."
            ),
        ),
        QuantizationTargetPolicyDecision(
            kind="tensor",
            names=later_value_tensors,
            reason=(
                "The v0/v1/v2 tensors gate every later layer's dependency on "
                "v_first and remain high precision in every candidate."
            ),
        ),
    ]
    if protection_profile in {"critical-high", "low-rank-w8-critical-high"}:
        protections.append(
            QuantizationTargetPolicyDecision(
                kind="module",
                names=other_time_mix_modules,
                reason=(
                    "Keep the remaining TimeMix projections high precision in "
                    "critical-high candidates."
                ),
            )
        )
    protections.extend(
        [
            QuantizationTargetPolicyDecision(
                kind="module",
                names=[_HEAD_IGNORE],
                reason="Keep the output head high precision in every candidate.",
            ),
            QuantizationTargetPolicyDecision(
                kind="tensor",
                names=recurrent_time_mix_tensors,
                reason=(
                    "Keep all unselected TimeMix recurrent Parameters high "
                    "precision; v0/v1/v2 remain on the protected v_first path."
                ),
            ),
        ]
    )
    metadata = QuantizationTargetPolicyMetadata(
        protection_profile=protection_profile,
        base_model_prefix=base_model_prefix,
        selection=QuantizationTargetPolicyDecision(
            kind="module",
            names=selected_modules,
            reason=(
                "Select standard RWKV-7 ChannelMix projections and, for the "
                "v-first-dataflow ablation only, non-value TimeMix projections; "
                "the complete v_first value path remains high precision."
            ),
        ),
        parameter_selection=(
            QuantizationTargetPolicyDecision(
                kind="tensor",
                names=selected_low_rank_tensors,
                reason=(
                    "Quantize only standard raw w/a/g low-rank Parameters to W8; "
                    "v1/v2 and the layer-0 value projection remain high precision."
                ),
            )
            if selected_low_rank_tensors
            else None
        ),
        protections=protections,
    )
    return policy_ignore, metadata
