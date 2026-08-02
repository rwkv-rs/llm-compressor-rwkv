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
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import unquote, urlparse

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "QuantizationTargetPolicyDecision",
    "QuantizationTargetPolicyMetadata",
    "RWKV7ArtifactContract",
    "RWKV7CheckpointContract",
    "RWKV7ImplementationProvenance",
    "RWKV7QuantizationRecipeMetadata",
    "RWKV7RepositoryContract",
    "RWKV7TransformersProvenance",
    "apply_rwkv7_target_policy",
    "audit_rwkv7_quantized_checkpoint",
    "build_rwkv7_artifact_contract",
    "build_rwkv7_quantization_recipe",
    "quantize_rwkv7_oneshot",
    "run_rwkv7_checkpoint_candidate",
    "validate_rwkv7_transformers_provenance",
    "validate_rwkv7_implementation_provenance",
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
    "a0",
    "k_k",
    "k_a",
    "r_k",
)
_LOW_RANK_LINEAR_NAMES = ("w1", "w2", "a1", "a2", "g1", "g2")
_V_FIRST_LINEAR_NAMES = ("v1", "v2")
_VLLM_NVFP4_LOW_RANK_NAMES = (
    "w1",
    "w2",
    "a1",
    "a2",
    "v1",
    "v2",
    "g1",
    "g2",
)
_SUPPORTED_FRAMEWORK_VERSIONS = {
    "compressed_tensors": "0.17.2.a20260731",
    "transformers": "5.15.0.dev0",
}
_CANDIDATE_SPECS = {
    "nvfp4-w4a4": {
        "scheme": "NVFP4",
        "protection_profile": "critical-high",
        "candidate_role": "nvfp4-primary",
        "runtime_requirement": "blackwell-sm120",
    },
    "nvfp4-w4a16": {
        "scheme": "NVFP4A16",
        "protection_profile": "critical-high",
        "candidate_role": "nvfp4-weight-only-baseline",
        "runtime_requirement": "blackwell-sm120",
    },
    "nvfp4-w4a16-protection-ablation": {
        "scheme": "NVFP4A16",
        "protection_profile": "v-first-dataflow",
        "candidate_role": "nvfp4-protection-ablation",
        "runtime_requirement": "blackwell-sm120",
    },
    "w8a16-low-rank-critical-high": {
        "scheme": "W8A16",
        "protection_profile": "low-rank-w8-critical-high",
        "candidate_role": "low-rank-w8-diagnostic",
        "runtime_requirement": "portable-int8",
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
_LLM_COMPRESSOR_FORK_REPOSITORY = "https://github.com/rwkv-rs/llm-compressor-rwkv.git"
_TRANSFORMERS_RWKV_REPOSITORY = "https://github.com/rwkv-rs/transformers-rwkv.git"
_TRANSFORMERS_RWKV_OID = "5d11fbe2559fec5611798bd6cc3f6c89ae145f68"
_RWKV7_CANONICAL_REPOSITORIES = frozenset(
    {
        "vllm-project/llm-compressor",
        "rwkv-rs/llm-compressor-rwkv",
        "rwkv-rs/transformers-rwkv",
    }
)
_VLLM_RWKV_NVFP4_W4A16_CONSUMER_REVISION = "88b992bbc73e8b904ae672dfd39396b6dd0d6ea4"
_TRANSFORMERS_RWKV_CONSUMER = "transformers-rwkv-compressed-tensors"
_VLLM_RWKV_NVFP4_W4A4_CONSUMER = "vllm-rwkv-nvfp4-w4a4"
_VLLM_RWKV_NVFP4_W4A16_CONSUMER = "vllm-rwkv-nvfp4-w4a16"
_G1H_1_5B_CHECKPOINT = {
    "model_id": "g1h-1.5b",
    "repository": "BlinkDL/rwkv7-g1",
    "revision": "6d5762253b343eec6cfbf5ed62f872f30a4cd89c",
    "filename": "rwkv7-g1h-1.5b-20260710-ctx10240.pth",
    "sha256": "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c",
    "size_bytes": 3055444605,
}


def _vllm_nvfp4_protection_ablation_targets(
    base_model_prefix: str,
    num_hidden_layers: int,
) -> list[str]:
    if num_hidden_layers < 2:
        raise ValueError("RWKV-7 vLLM protection ablation requires at least two blocks")
    layer_zero_prefix = f"{base_model_prefix}.blocks.0.att"
    targets = [
        *(f"{layer_zero_prefix}.{name}" for name in _LOW_RANK_LINEAR_NAMES),
        *(f"{layer_zero_prefix}.{name}" for name in ("receptance", "key", "output")),
    ]
    targets.extend(
        target
        for layer_id in range(1, num_hidden_layers)
        for target in (
            *(
                f"{base_model_prefix}.blocks.{layer_id}.att.{name}"
                for name in _VLLM_NVFP4_LOW_RANK_NAMES
            ),
            *(
                f"{base_model_prefix}.blocks.{layer_id}.att.{name}"
                for name in _TIME_MIX_LINEAR_NAMES
            ),
        )
    )
    return targets


def _critical_high_targets(
    base_model_prefix: str,
    num_hidden_layers: int,
) -> list[str]:
    return [
        f"{base_model_prefix}.blocks.{layer_id}.ffn.{name}"
        for layer_id in range(num_hidden_layers)
        for name in ("key", "value")
    ]


def _low_rank_w8_critical_high_targets(
    base_model_prefix: str,
    num_hidden_layers: int,
) -> list[str]:
    return [
        *(
            f"{base_model_prefix}.blocks.{layer_id}.att.{name}"
            for layer_id in range(num_hidden_layers)
            for name in _LOW_RANK_LINEAR_NAMES
        ),
        *_critical_high_targets(base_model_prefix, num_hidden_layers),
    ]


def _exact_target_schema_modules(
    target_schema: str,
    base_model_prefix: str,
    num_hidden_layers: int,
) -> list[str]:
    if target_schema == "rwkv7-nvfp4-critical-high-v1":
        return _critical_high_targets(base_model_prefix, num_hidden_layers)
    if target_schema == "rwkv7-nvfp4-protection-ablation-no-ffn-v1":
        return _vllm_nvfp4_protection_ablation_targets(
            base_model_prefix,
            num_hidden_layers,
        )
    if target_schema == "rwkv7-w8-low-rank-critical-high-v1":
        return _low_rank_w8_critical_high_targets(
            base_model_prefix,
            num_hidden_layers,
        )
    raise ValueError(f"unsupported RWKV-7 target schema: {target_schema}")


def _target_fqns_digest(names: list[str]) -> str:
    serialized = json.dumps(names, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class RWKV7RepositoryContract(BaseModel):
    """Immutable source/fork identity for this RWKV-7 adaptation."""

    model_config = ConfigDict(extra="forbid")

    upstream_repository: Literal[
        "https://github.com/vllm-project/llm-compressor.git"
    ] = _LLM_COMPRESSOR_UPSTREAM_REPOSITORY
    upstream_oid: Literal["28c9c76b74cdd47076f95d012227482d22a8f365"] = (
        _LLM_COMPRESSOR_UPSTREAM_OID
    )
    fork_repository: Literal["https://github.com/rwkv-rs/llm-compressor-rwkv.git"] = (
        _LLM_COMPRESSOR_FORK_REPOSITORY
    )
    transformers_repository: Literal[
        "https://github.com/rwkv-rs/transformers-rwkv.git"
    ] = _TRANSFORMERS_RWKV_REPOSITORY
    transformers_oid: Literal["5d11fbe2559fec5611798bd6cc3f6c89ae145f68"] = (
        _TRANSFORMERS_RWKV_OID
    )


class RWKV7TransformersProvenance(BaseModel):
    """Observed Transformers and delegated operator-runtime provenance."""

    model_config = ConfigDict(extra="forbid")

    repository: str
    revision: str
    installation_source: Literal["pep610-vcs", "editable-git"]
    editable: bool
    operator_runtime: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_revision(self) -> RWKV7TransformersProvenance:
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("RWKV-7 Transformers provenance requires a full Git OID")
        return self


class RWKV7ImplementationProvenance(BaseModel):
    """Observed editable llm-compressor fork implementation identity."""

    model_config = ConfigDict(extra="forbid")

    repository: str
    revision: str
    installation_source: Literal["editable-git"] = "editable-git"
    editable: Literal[True] = True

    @model_validator(mode="after")
    def validate_revision(self) -> RWKV7ImplementationProvenance:
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("RWKV-7 llm-compressor provenance requires a full Git OID")
        return self


def _normalize_operator_runtime_provenance(
    provenance: object,
) -> dict[str, str]:
    if not isinstance(provenance, Mapping):
        raise RuntimeError(
            "RWKV-7 Transformers public runtime provenance gate returned "
            "non-mapping evidence"
        )
    normalized = {}
    for key, value in provenance.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise RuntimeError(
                "RWKV-7 Transformers public runtime provenance evidence must "
                "contain only string keys and values"
            )
        normalized[key] = value
    required = {
        "distribution",
        "distribution_version",
        "extra",
        "flash_rwkv_distribution",
        "flash_rwkv_distribution_version",
        "flash_rwkv_repository",
        "flash_rwkv_revision",
        "flash_rwkv_source_kind",
        "repository",
        "requirement",
        "revision",
        "source_kind",
    }
    missing = sorted(required - normalized.keys())
    if missing:
        raise RuntimeError(
            "RWKV-7 Transformers public runtime provenance evidence is incomplete: "
            f"missing={missing}"
        )
    return dict(sorted(normalized.items()))


def _canonical_repository_url(repository: str) -> str:
    if not isinstance(repository, str) or not repository:
        raise RuntimeError("RWKV-7 repository URL must be a non-empty string")
    if any(
        ord(character) <= 0x20 or ord(character) >= 0x7F for character in repository
    ):
        raise RuntimeError(
            "RWKV-7 repository URL must contain only visible ASCII characters"
        )
    if "%" in repository:
        raise RuntimeError("RWKV-7 repository URL must not contain percent encoding")

    normalized = repository.removeprefix("git+")
    parsed = urlparse(normalized)
    try:
        port = parsed.port
    except ValueError as error:
        raise RuntimeError("RWKV-7 repository URL has an invalid port") from error
    if parsed.scheme.lower() != "https":
        raise RuntimeError("RWKV-7 repository URL must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("RWKV-7 repository URL must not contain userinfo")
    if port is not None or parsed.netloc.lower() != "github.com":
        raise RuntimeError("RWKV-7 repository URL must use github.com without a port")
    if parsed.params or parsed.query or parsed.fragment:
        raise RuntimeError(
            "RWKV-7 repository URL must not contain parameters, query, or fragment"
        )
    if "//" in parsed.path:
        raise RuntimeError("RWKV-7 repository URL must not contain repeated slashes")

    path = parsed.path[:-1] if parsed.path.endswith("/") else parsed.path
    parts = path.removeprefix("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(
            "RWKV-7 repository URL must contain exactly one owner and repository"
        )
    owner, name = parts
    if name.lower().endswith(".git"):
        name = name[:-4]
        if name.lower().endswith(".git"):
            raise RuntimeError(
                "RWKV-7 repository URL must contain at most one .git suffix"
            )
    github_name = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
    if github_name.fullmatch(owner) is None or github_name.fullmatch(name) is None:
        raise RuntimeError("RWKV-7 repository URL contains an invalid GitHub path")

    canonical_path = f"{owner.lower()}/{name.lower()}"
    if canonical_path not in _RWKV7_CANONICAL_REPOSITORIES:
        raise RuntimeError(
            "RWKV-7 repository URL is not one of the exact contract repositories"
        )
    return f"https://github.com/{canonical_path}"


def _git_provenance_value(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(
            "RWKV-7 exact Transformers provenance requires a working Git "
            "executable for editable installs"
        ) from error
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            "RWKV-7 could not resolve exact Transformers Git provenance: "
            f"git {' '.join(arguments)} failed"
        )
    return result.stdout.strip()


def _installed_transformers_provenance() -> RWKV7TransformersProvenance:
    requirement = (
        "RWKV-7 artifact/export/load requires Transformers installed from an "
        "exact rwkv-rs/transformers-rwkv VCS revision"
    )
    try:
        distribution = importlib_metadata.distribution("transformers")
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"{requirement}; distribution metadata is missing"
        ) from error
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(
            f"{requirement}; registry-only installation has no PEP 610 provenance"
        )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{requirement}; direct_url.json is invalid") from error

    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, dict) and vcs_info.get("vcs") == "git":
        revision = vcs_info.get("commit_id")
        requested_revision = vcs_info.get("requested_revision")
        repository = direct_url.get("url")
        if (
            not isinstance(revision, str)
            or not isinstance(requested_revision, str)
            or not isinstance(repository, str)
        ):
            raise RuntimeError(f"{requirement}; PEP 610 VCS metadata is incomplete")
        if requested_revision != revision:
            raise RuntimeError(
                f"{requirement}; requested revision is not exact resolved OID"
            )

        import transformers

        module_path = Path(transformers.__file__).resolve()
        distribution_module = Path(
            distribution.locate_file("transformers/__init__.py")
        ).resolve()
        if module_path != distribution_module:
            raise RuntimeError(
                f"{requirement}; imported module does not belong to its distribution"
            )
        return RWKV7TransformersProvenance(
            repository=repository,
            revision=revision,
            installation_source="pep610-vcs",
            editable=False,
        )

    directory_info = direct_url.get("dir_info")
    source_url = direct_url.get("url")
    if (
        not isinstance(directory_info, dict)
        or directory_info.get("editable") is not True
        or not isinstance(source_url, str)
    ):
        raise RuntimeError(f"{requirement}; PEP 610 metadata is not exact VCS data")
    parsed_source = urlparse(source_url)
    if parsed_source.scheme != "file" or parsed_source.netloc not in ("", "localhost"):
        raise RuntimeError(f"{requirement}; editable source is not a local file URL")
    repository_root = Path(unquote(parsed_source.path)).resolve()
    if not repository_root.is_dir():
        raise RuntimeError(f"{requirement}; editable source directory is missing")

    import transformers

    module_path = Path(transformers.__file__).resolve()
    repository_top_level = Path(
        _git_provenance_value(repository_root, "rev-parse", "--show-toplevel")
    ).resolve()
    if repository_top_level != repository_root:
        raise RuntimeError(
            f"{requirement}; editable source is not the Git repository root"
        )
    expected_module_path = repository_root / "src/transformers/__init__.py"
    if module_path != expected_module_path.resolve():
        raise RuntimeError(
            f"{requirement}; imported module does not belong to the editable "
            "Transformers source package"
        )
    revision = _git_provenance_value(repository_root, "rev-parse", "HEAD")
    repository = _git_provenance_value(
        repository_root,
        "remote",
        "get-url",
        "origin",
    )
    dirty = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if dirty.returncode != 0:
        raise RuntimeError(f"{requirement}; editable Git status could not be read")
    if dirty.stdout.strip():
        raise RuntimeError(f"{requirement}; editable source is dirty")
    return RWKV7TransformersProvenance(
        repository=repository,
        revision=revision,
        installation_source="editable-git",
        editable=True,
    )


def _validate_rwkv7_transformers_source_provenance(
    contract: RWKV7RepositoryContract | None = None,
) -> RWKV7TransformersProvenance:
    """Fail closed unless Transformers comes from the exact self-owned revision."""

    expected = RWKV7RepositoryContract() if contract is None else contract
    observed = _installed_transformers_provenance()
    repository_matches = _canonical_repository_url(
        observed.repository
    ) == _canonical_repository_url(expected.transformers_repository)
    if not repository_matches or observed.revision != expected.transformers_oid:
        raise RuntimeError(
            "RWKV-7 Transformers provenance mismatch: "
            f"expected={expected.transformers_repository}@{expected.transformers_oid} "
            f"actual={observed.repository}@{observed.revision}"
        )
    return observed


def validate_rwkv7_transformers_provenance(
    contract: RWKV7RepositoryContract | None = None,
) -> RWKV7TransformersProvenance:
    """Validate the self-owned Transformers source and delegated operator runtime."""

    observed = _validate_rwkv7_transformers_source_provenance(contract)
    from transformers.models.rwkv7 import validate_rwkv7_runtime_provenance

    if not callable(validate_rwkv7_runtime_provenance):
        raise RuntimeError(
            "RWKV-7 Transformers fork lacks its public runtime provenance gate"
        )
    operator_runtime = _normalize_operator_runtime_provenance(
        validate_rwkv7_runtime_provenance()
    )
    return observed.model_copy(update={"operator_runtime": operator_runtime})


def validate_rwkv7_implementation_provenance(
    implementation_revision: str,
    contract: RWKV7RepositoryContract | None = None,
) -> RWKV7ImplementationProvenance:
    """Bind formal candidate evidence to this clean llm-compressor checkout."""

    if re.fullmatch(r"[0-9a-f]{40}", implementation_revision) is None:
        raise ValueError("implementation_revision must be a full lowercase Git OID")
    expected = RWKV7RepositoryContract() if contract is None else contract
    requirement = (
        "formal RWKV-7 execution requires a clean editable "
        "rwkv-rs/llm-compressor-rwkv checkout"
    )
    try:
        distribution = importlib_metadata.distribution("llmcompressor")
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"{requirement}; distribution metadata is missing"
        ) from error
    if distribution.metadata.get("Name") != "llmcompressor":
        raise RuntimeError(f"{requirement}; distribution identity is invalid")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(f"{requirement}; editable PEP 610 metadata is missing")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{requirement}; direct_url.json is invalid") from error
    directory_info = direct_url.get("dir_info")
    parsed_source = urlparse(str(direct_url.get("url", "")))
    if (
        not isinstance(directory_info, dict)
        or directory_info.get("editable") is not True
        or parsed_source.scheme != "file"
        or parsed_source.netloc not in ("", "localhost")
    ):
        raise RuntimeError(f"{requirement}; installation is not a local editable")
    repository_root = Path(unquote(parsed_source.path)).resolve()
    if not repository_root.is_dir():
        raise RuntimeError(f"{requirement}; editable source directory is missing")

    import llmcompressor

    module_path = Path(llmcompressor.__file__).resolve()
    expected_module_path = repository_root / "src/llmcompressor/__init__.py"
    if module_path != expected_module_path.resolve():
        raise RuntimeError(
            f"{requirement}; imported module does not belong to the editable source"
        )
    try:
        repository_top_level = Path(
            _git_provenance_value(repository_root, "rev-parse", "--show-toplevel")
        ).resolve()
    except RuntimeError as git_error:
        repository = os.environ.get("LLMCOMPRESSOR_RWKV_REPOSITORY", "")
        revision = os.environ.get("LLMCOMPRESSOR_RWKV_REVISION", "")
        if not repository or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise RuntimeError(
                f"{requirement}; gitless managed exports require exact repository "
                "and revision provenance"
            ) from git_error
    else:
        if repository_top_level != repository_root:
            raise RuntimeError(f"{requirement}; editable source is not the Git root")
        repository = _git_provenance_value(
            repository_root,
            "remote",
            "get-url",
            "origin",
        )
        revision = _git_provenance_value(repository_root, "rev-parse", "HEAD")
        dirty = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        if dirty.returncode != 0:
            raise RuntimeError(f"{requirement}; Git status could not be read")
        if dirty.stdout.strip():
            raise RuntimeError(f"{requirement}; editable source is dirty")
    if _canonical_repository_url(repository) != _canonical_repository_url(
        expected.fork_repository
    ):
        raise RuntimeError(
            "RWKV-7 llm-compressor repository provenance mismatch: "
            f"expected={expected.fork_repository} actual={repository}"
        )
    if revision != implementation_revision:
        raise RuntimeError(
            "RWKV-7 implementation revision differs from the active checkout: "
            f"expected={implementation_revision} actual={revision}"
        )
    return RWKV7ImplementationProvenance(
        repository=repository,
        revision=revision,
    )


class RWKV7CheckpointContract(BaseModel):
    """Pinned real 1.5B source checkpoint and standard conversion contract."""

    model_config = ConfigDict(extra="forbid")

    model_id: Literal["g1h-1.5b"] = _G1H_1_5B_CHECKPOINT["model_id"]
    repository: Literal["BlinkDL/rwkv7-g1"] = _G1H_1_5B_CHECKPOINT["repository"]
    revision: Literal["6d5762253b343eec6cfbf5ed62f872f30a4cd89c"] = (
        _G1H_1_5B_CHECKPOINT["revision"]
    )
    filename: Literal["rwkv7-g1h-1.5b-20260710-ctx10240.pth"] = _G1H_1_5B_CHECKPOINT[
        "filename"
    ]
    sha256: Literal[
        "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c"
    ] = _G1H_1_5B_CHECKPOINT["sha256"]
    size_bytes: Literal[3055444605] = _G1H_1_5B_CHECKPOINT["size_bytes"]
    source_format: Literal["legacy_pth"] = "legacy_pth"
    converted_format: Literal["standard_hf_safetensors"] = "standard_hf_safetensors"
    architecture: Literal["Rwkv7ForCausalLM"] = "Rwkv7ForCausalLM"
    model_type: Literal["rwkv7"] = "rwkv7"
    embedding_layer_norm_fused: Literal[False] = False


class RWKV7VLLMLoaderMetadata(BaseModel):
    """Producer inventory and executable standard-HF consumer capabilities."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["Rwkv7ForCausalLM"] = "Rwkv7ForCausalLM"
    model_type: Literal["rwkv7"] = "rwkv7"
    source_format: Literal["standard_hf"] = "standard_hf"
    load_format: Literal["safetensors"] = "safetensors"
    quant_method: Literal["compressed-tensors"] = "compressed-tensors"
    quantization_format: Literal["nvfp4-pack-quantized", "pack-quantized"]
    target_schema_version: Literal[1] = 1
    target_schema: Literal[
        "rwkv7-nvfp4-critical-high-v1",
        "rwkv7-nvfp4-protection-ablation-no-ffn-v1",
        "rwkv7-w8-low-rank-critical-high-v1",
    ]
    num_hidden_layers: int = Field(ge=1)
    quantized_target_fqns_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumer_capabilities: list[
        Literal[
            "transformers-rwkv-compressed-tensors",
            "vllm-rwkv-nvfp4-w4a4",
            "vllm-rwkv-nvfp4-w4a16",
        ]
    ]
    vllm_consumer_requirement: (
        Literal[
            "vllm-rwkv-nvfp4-w4a4",
            "vllm-rwkv-nvfp4-w4a16",
        ]
        | None
    )
    vllm_consumer_revision: str | None
    embedding_name: Literal["model.embeddings.weight"] = "model.embeddings.weight"
    block_prefix: Literal["model.blocks."] = "model.blocks."
    output_norm_prefix: Literal["model.ln_out."] = "model.ln_out."
    head_name: Literal["head.weight"] = "head.weight"
    legacy_pth_direct_load: Literal[False] = False
    quantization_target_type: Literal["Linear"] = "Linear"
    linear_weight_suffix: Literal["weight"] = "weight"
    linear_weight_layout: Literal["out-in"] = "out-in"
    quantized_modules: list[str]
    quantized_weight_names: list[str]
    low_rank_linear_modules: list[str]
    quantized_low_rank_modules: list[str]
    protected_v_first_linear_modules: list[str]
    layer_zero_v_first_producer: str
    protected_linear_modules: list[str]
    protected_embedding_modules: list[str]
    protected_normalization_modules: list[str]
    protected_state_tensors: list[str]
    protected_modules: list[str]
    protected_tensors: list[str]
    protected_parameter_keys: list[str]

    @model_validator(mode="after")
    def validate_standard_linear_ownership(self) -> RWKV7VLLMLoaderMetadata:
        inventories = {
            "quantized_modules": self.quantized_modules,
            "quantized_weight_names": self.quantized_weight_names,
            "low_rank_linear_modules": self.low_rank_linear_modules,
            "quantized_low_rank_modules": self.quantized_low_rank_modules,
            "protected_v_first_linear_modules": (self.protected_v_first_linear_modules),
            "protected_linear_modules": self.protected_linear_modules,
            "protected_embedding_modules": self.protected_embedding_modules,
            "protected_normalization_modules": (self.protected_normalization_modules),
            "protected_state_tensors": self.protected_state_tensors,
            "protected_modules": self.protected_modules,
            "protected_tensors": self.protected_tensors,
            "protected_parameter_keys": self.protected_parameter_keys,
        }
        for label, names in inventories.items():
            if len(names) != len(set(names)):
                raise ValueError(f"RWKV-7 vLLM metadata duplicates {label}")
        if len(self.consumer_capabilities) != len(set(self.consumer_capabilities)):
            raise ValueError("RWKV-7 consumer capabilities must be unique")
        expected_weight_names = [
            f"{name}.{self.linear_weight_suffix}" for name in self.quantized_modules
        ]
        if self.quantized_weight_names != expected_weight_names:
            raise ValueError(
                "RWKV-7 vLLM quantized weights must use standard Linear ownership"
            )
        if self.quantized_target_fqns_digest != _target_fqns_digest(
            self.quantized_modules
        ):
            raise ValueError(
                "RWKV-7 quantized target FQN digest differs from the exact inventory"
            )
        quantized = set(self.quantized_modules)
        protected = set(self.protected_modules)
        protected_linear = set(self.protected_linear_modules)
        protected_embedding = set(self.protected_embedding_modules)
        protected_normalization = set(self.protected_normalization_modules)
        inventory_block_ids = sorted(
            {
                int(match.group(1))
                for name in [
                    *self.quantized_modules,
                    *self.protected_modules,
                    *self.protected_state_tensors,
                ]
                if (match := re.match(r"^model\.blocks\.(\d+)\.", name))
            }
        )
        if inventory_block_ids != list(range(self.num_hidden_layers)):
            raise ValueError(
                "RWKV-7 layer count differs from the complete module/tensor inventory"
            )
        expected_quantized_modules = _exact_target_schema_modules(
            self.target_schema,
            "model",
            self.num_hidden_layers,
        )
        if self.quantized_modules != expected_quantized_modules:
            raise ValueError(
                "RWKV-7 quantized modules differ from the exact expanded target "
                f"schema {self.target_schema!r}"
            )
        low_rank = set(self.low_rank_linear_modules)
        quantized_low_rank = set(self.quantized_low_rank_modules)
        protected_v_first = set(self.protected_v_first_linear_modules)
        if quantized & protected:
            raise ValueError("RWKV-7 vLLM quantized and protected modules overlap")
        if (
            protected_linear | protected_embedding | protected_normalization
        ) != protected:
            raise ValueError("RWKV-7 vLLM protected module categories are incomplete")
        if (
            protected_linear & protected_embedding
            or protected_linear & protected_normalization
            or protected_embedding & protected_normalization
        ):
            raise ValueError("RWKV-7 vLLM protected module categories must be disjoint")
        if self.protected_state_tensors != self.protected_tensors:
            raise ValueError("RWKV-7 vLLM protected state inventory drifted")
        expected_protected_parameter_keys = [
            *(f"{name}.weight" for name in self.protected_linear_modules),
            *(f"{name}.weight" for name in self.protected_embedding_modules),
            *(
                parameter_name
                for name in self.protected_normalization_modules
                for parameter_name in (f"{name}.weight", f"{name}.bias")
            ),
            *self.protected_state_tensors,
        ]
        if self.protected_parameter_keys != expected_protected_parameter_keys:
            raise ValueError(
                "RWKV-7 protected parameter inventory must include every "
                "Linear/Embedding weight, normalization weight and bias, and "
                "protected state tensor"
            )
        if not quantized_low_rank <= low_rank or not quantized_low_rank <= quantized:
            raise ValueError(
                "RWKV-7 vLLM quantized low-rank modules drifted from ownership"
            )
        if not low_rank <= quantized | protected:
            raise ValueError("RWKV-7 vLLM low-rank module inventory is incomplete")
        if not protected_v_first <= protected or protected_v_first & quantized:
            raise ValueError("RWKV-7 vLLM v_first Linear protection is incomplete")
        if self.layer_zero_v_first_producer not in protected_linear:
            raise ValueError("RWKV-7 vLLM layer-0 v_first producer is not protected")
        if self.quantization_format == "pack-quantized":
            if self.consumer_capabilities != [_TRANSFORMERS_RWKV_CONSUMER]:
                raise ValueError(
                    "RWKV-7 W8 pack-quantized artifacts are not executable vLLM "
                    "NVFP4 consumer artifacts"
                )
            if (
                self.vllm_consumer_requirement is not None
                or self.vllm_consumer_revision is not None
            ):
                raise ValueError(
                    "RWKV-7 W8 pack-quantized artifacts cannot claim a vLLM "
                    "consumer requirement or revision"
                )
        else:
            if self.vllm_consumer_requirement not in {
                _VLLM_RWKV_NVFP4_W4A4_CONSUMER,
                _VLLM_RWKV_NVFP4_W4A16_CONSUMER,
            }:
                raise ValueError(
                    "RWKV-7 NVFP4 artifacts must declare an exact W4A4 or W4A16 "
                    "vLLM consumer requirement"
                )
            has_executable_capability = (
                self.vllm_consumer_requirement in self.consumer_capabilities
            )
            if has_executable_capability != (self.vllm_consumer_revision is not None):
                raise ValueError(
                    "RWKV-7 NVFP4 executable capability and consumer revision "
                    "must be declared together"
                )
            if (
                self.vllm_consumer_revision is not None
                and re.fullmatch(r"[0-9a-f]{40}", self.vllm_consumer_revision) is None
            ):
                raise ValueError(
                    "RWKV-7 NVFP4 vLLM consumer revision must be a full Git OID"
                )
        return self


class RWKV7ArtifactContract(BaseModel):
    """Self-contained loader and protection contract serialized with a result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = 3
    repository: RWKV7RepositoryContract
    runtime_provenance: RWKV7TransformersProvenance
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

    @model_validator(mode="after")
    def validate_applied_quantization(self) -> RWKV7ArtifactContract:
        recipe = self.target_policy.recipe
        if recipe is None or not recipe.quantization_applied:
            raise ValueError(
                "RWKV-7 compressed artifact metadata must record applied quantization"
            )
        if self.candidate != recipe.candidate:
            raise ValueError("RWKV-7 artifact candidate must match recipe metadata")
        if self.runtime_provenance != recipe.runtime_provenance:
            raise ValueError(
                "RWKV-7 artifact runtime provenance must match recipe metadata"
            )
        expected_format = (
            "pack-quantized"
            if self.candidate == "w8a16-low-rank-critical-high"
            else "nvfp4-pack-quantized"
        )
        if self.vllm.quantization_format != expected_format:
            raise ValueError(
                "RWKV-7 artifact candidate and quantization format are inconsistent"
            )
        expected_target_schema = {
            "nvfp4-w4a4": "rwkv7-nvfp4-critical-high-v1",
            "nvfp4-w4a16": "rwkv7-nvfp4-critical-high-v1",
            "nvfp4-w4a16-protection-ablation": (
                "rwkv7-nvfp4-protection-ablation-no-ffn-v1"
            ),
            "w8a16-low-rank-critical-high": ("rwkv7-w8-low-rank-critical-high-v1"),
        }[self.candidate]
        if self.vllm.target_schema != expected_target_schema:
            raise ValueError(
                "RWKV-7 artifact candidate and target schema are inconsistent"
            )
        expected_vllm_requirement = {
            "nvfp4-w4a4": _VLLM_RWKV_NVFP4_W4A4_CONSUMER,
            "nvfp4-w4a16": _VLLM_RWKV_NVFP4_W4A16_CONSUMER,
            "nvfp4-w4a16-protection-ablation": (_VLLM_RWKV_NVFP4_W4A16_CONSUMER),
            "w8a16-low-rank-critical-high": None,
        }[self.candidate]
        if self.vllm.vllm_consumer_requirement != expected_vllm_requirement:
            raise ValueError(
                "RWKV-7 artifact candidate and exact vLLM consumer requirement "
                "are inconsistent"
            )
        if self.candidate == "nvfp4-w4a4":
            if (
                self.vllm.consumer_capabilities != [_TRANSFORMERS_RWKV_CONSUMER]
                or self.vllm.vllm_consumer_revision is not None
            ):
                raise ValueError(
                    "RWKV-7 W4A4 cannot claim vLLM execution before its native "
                    "consumer capability is revision-pinned"
                )
        elif self.candidate in {
            "nvfp4-w4a16",
            "nvfp4-w4a16-protection-ablation",
        }:
            if (
                self.vllm.consumer_capabilities
                != [
                    _TRANSFORMERS_RWKV_CONSUMER,
                    _VLLM_RWKV_NVFP4_W4A16_CONSUMER,
                ]
                or self.vllm.vllm_consumer_revision
                != _VLLM_RWKV_NVFP4_W4A16_CONSUMER_REVISION
            ):
                raise ValueError(
                    "RWKV-7 W4A16 requires its revision-pinned vLLM consumer capability"
                )
        if self.vllm.quantized_modules != self.target_policy.selection.names:
            raise ValueError(
                "RWKV-7 artifact quantized inventory must match target policy"
            )
        if self.candidate == "nvfp4-w4a16-protection-ablation":
            expected_ablation_targets = _vllm_nvfp4_protection_ablation_targets(
                self.target_policy.base_model_prefix,
                self.vllm.num_hidden_layers,
            )
            if self.vllm.quantized_modules != expected_ablation_targets:
                raise ValueError(
                    "RWKV-7 protection ablation must exactly match the validated "
                    "no-FFN 9+12*(L-1) vLLM NVFP4 consumer matrix"
                )
            if len(expected_ablation_targets) != (
                9 + 12 * (self.vllm.num_hidden_layers - 1)
            ) or any(".ffn." in name for name in expected_ablation_targets):
                raise ValueError(
                    "RWKV-7 protection ablation target count or FFN exclusion drifted"
                )
        protected_modules = [
            name
            for decision in self.target_policy.protections
            if decision.kind == "module"
            for name in decision.names
        ]
        protected_tensors = [
            name
            for decision in self.target_policy.protections
            if decision.kind == "tensor"
            for name in decision.names
        ]
        if self.vllm.protected_modules != protected_modules:
            raise ValueError(
                "RWKV-7 artifact protected module inventory must match target policy"
            )
        if self.vllm.protected_tensors != protected_tensors:
            raise ValueError(
                "RWKV-7 artifact protected tensor inventory must match target policy"
            )
        if self.formal_checkpoint != (self.checkpoint is not None):
            raise ValueError(
                "RWKV-7 formal checkpoint flag must match checkpoint provenance"
            )
        return self


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
    candidate_role: Literal[
        "nvfp4-primary",
        "nvfp4-weight-only-baseline",
        "nvfp4-protection-ablation",
        "low-rank-w8-diagnostic",
    ]
    runtime_requirement: Literal["blackwell-sm120", "portable-int8"]
    algorithm: Literal["NVFP4", "INT8"]
    weight_dtype: Literal["float4", "int8"]
    weight_group_size: Literal[16, 32]
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
    runtime_provenance: RWKV7TransformersProvenance
    quantization_applied: bool = False

    @model_validator(mode="after")
    def validate_closed_contract(self):
        if self.candidate_order != list(_CANDIDATE_SCHEMES):
            raise ValueError(
                "RWKV-7 quantization candidate set or order is unsupported"
            )
        candidate_spec = _CANDIDATE_SPECS[self.candidate]
        if (
            self.candidate_role != candidate_spec["candidate_role"]
            or self.runtime_requirement != candidate_spec["runtime_requirement"]
        ):
            raise ValueError("RWKV-7 candidate role or runtime requirement drifted")
        _validate_framework_versions(self.framework_versions)
        if not self.targets:
            raise ValueError(
                "RWKV-7 quantization recipe requires resolved Linear targets"
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
    policy_version: Literal[2] = 2
    model_type: Literal["rwkv7"] = "rwkv7"
    protection_profile: Literal[
        "critical-high",
        "v-first-dataflow",
        "low-rank-w8-critical-high",
    ] = "critical-high"
    base_model_prefix: str
    selection: QuantizationTargetPolicyDecision
    protections: list[QuantizationTargetPolicyDecision]
    recipe: RWKV7QuantizationRecipeMetadata | None = None


def build_rwkv7_artifact_contract(
    target_policy: QuantizationTargetPolicyMetadata,
    candidate: str,
    *,
    checkpoint: RWKV7CheckpointContract | None = None,
    serialization_only_provenance: RWKV7TransformersProvenance | None = None,
) -> RWKV7ArtifactContract:
    """Resolve the exact standard-HF names protected at the runtime boundary."""

    if serialization_only_provenance is None:
        runtime_provenance = validate_rwkv7_transformers_provenance()
    else:
        runtime_provenance = _validate_rwkv7_transformers_source_provenance()
        if serialization_only_provenance != runtime_provenance:
            raise RuntimeError(
                "RWKV-7 serialization provenance differs from the active checkout"
            )
    if candidate not in _CANDIDATE_SCHEMES:
        raise ValueError(f"unsupported RWKV-7 candidate for artifact: {candidate}")
    if target_policy.recipe is None or target_policy.recipe.candidate != candidate:
        raise ValueError(
            "RWKV-7 artifact candidate must match resolved recipe metadata"
        )
    if target_policy.recipe.runtime_provenance != runtime_provenance:
        raise RuntimeError(
            "RWKV-7 recipe runtime provenance drifted before artifact export"
        )
    artifact_target_policy = target_policy.model_copy(
        update={
            "recipe": target_policy.recipe.model_copy(
                update={"quantization_applied": True}
            )
        }
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
    all_modules = [*target_policy.selection.names, *protected_modules]
    low_rank_suffixes = tuple(f".{name}" for name in _VLLM_NVFP4_LOW_RANK_NAMES)
    w_a_g_suffixes = tuple(f".{name}" for name in _LOW_RANK_LINEAR_NAMES)
    v_first_suffixes = tuple(f".{name}" for name in _V_FIRST_LINEAR_NAMES)
    low_rank_modules = sorted(
        name for name in all_modules if name.endswith(low_rank_suffixes)
    )
    quantized_low_rank_modules = [
        name
        for name in target_policy.selection.names
        if name.endswith(low_rank_suffixes)
    ]
    w_a_g_modules = {name for name in all_modules if name.endswith(w_a_g_suffixes)}
    protected_v_first_linear_modules = sorted(
        name for name in protected_modules if name.endswith(v_first_suffixes)
    )
    protected_linear_suffixes = tuple(
        f".{name}"
        for name in (
            *_TIME_MIX_LINEAR_NAMES,
            *_LOW_RANK_LINEAR_NAMES,
            *_V_FIRST_LINEAR_NAMES,
        )
    )
    protected_linear_modules = [
        name
        for name in protected_modules
        if name == _HEAD_IGNORE or name.endswith(protected_linear_suffixes)
    ]
    protected_embedding_modules = [
        name
        for name in protected_modules
        if name == f"{target_policy.base_model_prefix}.embeddings"
    ]
    protected_normalization_modules = [
        name
        for name in protected_modules
        if name not in set(protected_linear_modules) | set(protected_embedding_modules)
    ]
    protected_parameter_keys = [
        *(f"{name}.weight" for name in protected_linear_modules),
        *(f"{name}.weight" for name in protected_embedding_modules),
        *(
            parameter_name
            for name in protected_normalization_modules
            for parameter_name in (f"{name}.weight", f"{name}.bias")
        ),
        *protected_tensors,
    ]
    block_ids = sorted(
        {
            int(match.group(1))
            for name in [*all_modules, *protected_tensors]
            if (
                match := re.match(
                    rf"^{re.escape(target_policy.base_model_prefix)}\.blocks\."
                    r"(\d+)\.",
                    name,
                )
            )
        }
    )
    if not block_ids or block_ids != list(range(block_ids[-1] + 1)):
        raise ValueError(
            "RWKV-7 artifact block inventory must be non-empty and contiguous"
        )
    num_hidden_layers = block_ids[-1] + 1
    if (
        candidate == "w8a16-low-rank-critical-high"
        and set(quantized_low_rank_modules) != w_a_g_modules
    ):
        raise ValueError(
            "RWKV-7 low-rank W8 candidate must quantize every standard w/a/g "
            "Linear module"
        )
    if set(quantized_low_rank_modules) & set(protected_v_first_linear_modules):
        raise ValueError("RWKV-7 low-rank W8 selection touches v_first gating")
    layer_zero_value = f"{target_policy.base_model_prefix}.blocks.0.att.value"
    if layer_zero_value not in protected_modules:
        raise ValueError("RWKV-7 artifact must protect the layer-0 v_first producer")
    if any(name.startswith("rwkv7.") for name in target_policy.selection.names):
        raise ValueError("RWKV-7 artifact contains non-standard module names")
    return RWKV7ArtifactContract(
        repository=RWKV7RepositoryContract(),
        runtime_provenance=runtime_provenance,
        checkpoint=checkpoint,
        candidate=candidate,
        target_policy=artifact_target_policy,
        vllm=RWKV7VLLMLoaderMetadata(
            quantization_format=(
                "pack-quantized"
                if candidate == "w8a16-low-rank-critical-high"
                else "nvfp4-pack-quantized"
            ),
            target_schema=(
                "rwkv7-w8-low-rank-critical-high-v1"
                if candidate == "w8a16-low-rank-critical-high"
                else (
                    "rwkv7-nvfp4-protection-ablation-no-ffn-v1"
                    if candidate == "nvfp4-w4a16-protection-ablation"
                    else "rwkv7-nvfp4-critical-high-v1"
                )
            ),
            consumer_capabilities=(
                [_TRANSFORMERS_RWKV_CONSUMER]
                if candidate
                in {
                    "nvfp4-w4a4",
                    "w8a16-low-rank-critical-high",
                }
                else [
                    _TRANSFORMERS_RWKV_CONSUMER,
                    _VLLM_RWKV_NVFP4_W4A16_CONSUMER,
                ]
            ),
            vllm_consumer_requirement=(
                None
                if candidate == "w8a16-low-rank-critical-high"
                else (
                    _VLLM_RWKV_NVFP4_W4A4_CONSUMER
                    if candidate == "nvfp4-w4a4"
                    else _VLLM_RWKV_NVFP4_W4A16_CONSUMER
                )
            ),
            vllm_consumer_revision=(
                None
                if candidate
                in {
                    "nvfp4-w4a4",
                    "w8a16-low-rank-critical-high",
                }
                else _VLLM_RWKV_NVFP4_W4A16_CONSUMER_REVISION
            ),
            num_hidden_layers=num_hidden_layers,
            quantized_modules=list(target_policy.selection.names),
            quantized_target_fqns_digest=_target_fqns_digest(
                target_policy.selection.names
            ),
            quantized_weight_names=[
                f"{name}.weight" for name in target_policy.selection.names
            ],
            low_rank_linear_modules=low_rank_modules,
            quantized_low_rank_modules=quantized_low_rank_modules,
            protected_v_first_linear_modules=protected_v_first_linear_modules,
            layer_zero_v_first_producer=layer_zero_value,
            protected_linear_modules=protected_linear_modules,
            protected_embedding_modules=protected_embedding_modules,
            protected_normalization_modules=protected_normalization_modules,
            protected_state_tensors=protected_tensors,
            protected_modules=protected_modules,
            protected_tensors=protected_tensors,
            protected_parameter_keys=protected_parameter_keys,
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
    runtime_provenance: RWKV7TransformersProvenance,
) -> RWKV7QuantizationRecipeMetadata:
    inputs = scheme.input_activations
    is_w8 = candidate == "w8a16-low-rank-critical-high"
    actual_group = scheme.model_dump(mode="json")
    expected_group = _expected_quantization_group(candidate, serialized=False)
    if actual_group != expected_group:
        raise RuntimeError(
            "compressed-tensors scheme "
            f"{_CANDIDATE_SCHEMES[candidate]} drifted from the exact RWKV-7 "
            f"config group: expected={expected_group} actual={actual_group}"
        )

    return RWKV7QuantizationRecipeMetadata(
        candidate=candidate,
        candidate_order=list(_CANDIDATE_SCHEMES),
        candidate_role=_CANDIDATE_SPECS[candidate]["candidate_role"],
        runtime_requirement=_CANDIDATE_SPECS[candidate]["runtime_requirement"],
        algorithm="INT8" if is_w8 else "NVFP4",
        weight_dtype="int8" if is_w8 else "float4",
        weight_group_size=32 if is_w8 else 16,
        weight_scale_dtype="float32" if is_w8 else "float8_e4m3fn",
        input_dtype="float4" if inputs is not None else "float16",
        input_scale="dynamic_local" if inputs is not None else "none",
        input_scale_dtype="float8_e4m3fn" if inputs is not None else None,
        protection_profile=_CANDIDATE_SPECS[candidate]["protection_profile"],
        low_rank_weight_dtype="int8" if is_w8 else "none",
        targets=targets,
        framework_versions=framework_versions,
        runtime_provenance=runtime_provenance,
    )


def _build_rwkv7_quantization_recipe(
    model: torch.nn.Module,
    candidate: str,
    *,
    framework_versions: dict[str, str],
    runtime_provenance: RWKV7TransformersProvenance,
):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    from llmcompressor.modifiers.quantization import QuantizationModifier

    modifier_kwargs: dict[str, Any]
    if candidate == "w8a16-low-rank-critical-high":
        modifier_kwargs = {
            "config_groups": {
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=8,
                        type="int",
                        symmetric=True,
                        strategy="group",
                        group_size=32,
                    ),
                )
            }
        }
    else:
        modifier_kwargs = {"scheme": _CANDIDATE_SCHEMES[candidate]}
    modifier = QuantizationModifier(
        **modifier_kwargs,
        target_policy="rwkv7",
        target_policy_profile=_CANDIDATE_SPECS[candidate]["protection_profile"],
    )
    modifier._apply_target_policy(model)
    recipe_metadata = _validate_candidate_scheme(
        candidate,
        next(iter(modifier.resolved_config.config_groups.values())),
        targets=list(modifier.target_policy_metadata.selection.names),
        framework_versions=framework_versions,
        runtime_provenance=runtime_provenance,
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


def build_rwkv7_quantization_recipe(
    model: torch.nn.Module,
    candidate: str = "nvfp4-w4a4",
    *,
    framework_versions: dict[str, str] | None = None,
):
    """Build one operator-runtime validated closed-set quantization recipe."""

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
    return _build_rwkv7_quantization_recipe(
        model,
        candidate,
        framework_versions=versions,
        runtime_provenance=validate_rwkv7_transformers_provenance(),
    )


def _load_rwkv7_artifact_contract(output_dir: Path) -> RWKV7ArtifactContract:
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    try:
        return RWKV7ArtifactContract.model_validate(config.get(_RWKV7_METADATA_KEY))
    except ValueError as error:
        raise RuntimeError(
            "RWKV-7 serialized compressed artifact metadata is invalid"
        ) from error


def _expected_quantization_args(candidate: str) -> dict[str, Any]:
    if candidate == "w8a16-low-rank-critical-high":
        return {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": 32,
            "num_bits": 8,
            "observer": "memoryless_minmax",
            "observer_kwargs": {},
            "scale_dtype": None,
            "strategy": "group",
            "symmetric": True,
            "type": "int",
            "zp_dtype": None,
        }
    return {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": 16,
        "num_bits": 4,
        "observer": "memoryless_minmax",
        "observer_kwargs": {},
        "scale_dtype": "torch.float8_e4m3fn",
        "strategy": "tensor_group",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    }


def _expected_input_quantization_args(candidate: str) -> dict[str, Any] | None:
    if candidate != "nvfp4-w4a4":
        return None
    return {
        "actorder": None,
        "block_structure": None,
        "dynamic": "local",
        "group_size": 16,
        "num_bits": 4,
        "observer": "static_minmax",
        "observer_kwargs": {},
        "scale_dtype": "torch.float8_e4m3fn",
        "strategy": "tensor_group",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    }


def _expected_quantization_group(
    candidate: str,
    *,
    serialized: bool,
) -> dict[str, Any]:
    expected_format = (
        "pack-quantized"
        if candidate == "w8a16-low-rank-critical-high"
        else "nvfp4-pack-quantized"
    )
    return {
        "format": expected_format if serialized else None,
        "input_activations": _expected_input_quantization_args(candidate),
        "output_activations": None,
        "targets": ["Linear"],
        "weights": _expected_quantization_args(candidate),
    }


def _validate_native_rwkv7_config(serialized_config: Mapping[str, Any]) -> None:
    if serialized_config.get("model_type") != "rwkv7":
        raise RuntimeError("RWKV-7 artifact model_type must be exactly 'rwkv7'")
    if serialized_config.get("architectures") != ["Rwkv7ForCausalLM"]:
        raise RuntimeError(
            "RWKV-7 artifact architectures must be exactly ['Rwkv7ForCausalLM']"
        )
    if "auto_map" in serialized_config:
        raise RuntimeError(
            "RWKV-7 artifact must use native Transformers classes without auto_map"
        )


def _validate_native_rwkv7_runtime(config: Any, model: torch.nn.Module) -> None:
    from transformers import Rwkv7Config
    from transformers.models.rwkv7 import Rwkv7ForCausalLM

    if type(config) is not Rwkv7Config:
        raise RuntimeError(
            "RWKV-7 public load did not resolve the native Transformers Rwkv7Config"
        )
    if type(model) is not Rwkv7ForCausalLM:
        raise RuntimeError(
            "RWKV-7 public load did not resolve the native Transformers "
            "Rwkv7ForCausalLM"
        )


def _validate_rwkv7_loading_info(loading_info: Mapping[str, Any]) -> None:
    if not isinstance(loading_info, Mapping):
        raise RuntimeError("RWKV-7 public loader did not return loading information")
    for field in (
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    ):
        values = loading_info.get(field)
        if not isinstance(values, (list, tuple, set, frozenset)):
            raise RuntimeError(
                f"RWKV-7 public loader returned invalid {field}: {values!r}"
            )
        if values:
            raise RuntimeError(
                f"RWKV-7 public loader returned non-empty {field}: {list(values)!r}"
            )


def _expected_physical_tensor_keys(
    contract: RWKV7ArtifactContract,
) -> set[str]:
    expected = set(contract.vllm.protected_parameter_keys)
    for target in contract.vllm.quantized_modules:
        expected.update(
            {
                f"{target}.weight_packed",
                f"{target}.weight_scale",
            }
        )
        if contract.candidate == "w8a16-low-rank-critical-high":
            expected.add(f"{target}.weight_shape")
        else:
            expected.add(f"{target}.weight_global_scale")
            if contract.candidate == "nvfp4-w4a4":
                expected.add(f"{target}.input_global_scale")
    return expected


def _tensor_owners(tensors: dict[str, tuple[list[int], str]], suffix: str) -> set[str]:
    return {name.removesuffix(suffix) for name in tensors if name.endswith(suffix)}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor storage without retaining a second full-size model copy."""

    digest = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    chunk_elements = max(1, (8 * 1024 * 1024) // max(1, flat.element_size()))
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements]
        byte_view = chunk.contiguous().view(torch.uint8).cpu()
        digest.update(byte_view.numpy().tobytes())
    return digest.hexdigest()


def _validate_rwkv7_artifact_model_ownership(
    model: torch.nn.Module,
    contract: RWKV7ArtifactContract,
) -> dict[str, int]:
    """Resolve every artifact inventory entry against standard module ownership."""

    _resolve_standard_base_model(model)

    def require_module(
        name: str,
        expected_type: type[torch.nn.Module] | tuple[type[torch.nn.Module], ...],
    ) -> torch.nn.Module:
        try:
            module = model.get_submodule(name)
        except AttributeError as error:
            raise RuntimeError(
                f"RWKV-7 artifact metadata references a missing module: {name}"
            ) from error
        if not isinstance(module, expected_type):
            expected_names = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else "/".join(item.__name__ for item in expected_type)
            )
            raise RuntimeError(
                "RWKV-7 artifact module ownership drifted: "
                f"name={name} expected={expected_names} "
                f"actual={type(module).__name__}"
            )
        return module

    for name in contract.vllm.quantized_modules:
        require_module(name, torch.nn.Linear)
    for name in contract.vllm.protected_linear_modules:
        require_module(name, torch.nn.Linear)
    for name in contract.vllm.protected_embedding_modules:
        require_module(name, torch.nn.Embedding)
    for name in contract.vllm.protected_normalization_modules:
        require_module(name, (torch.nn.LayerNorm, torch.nn.GroupNorm))
    for name in contract.vllm.protected_state_tensors:
        try:
            parameter = model.get_parameter(name)
        except AttributeError as error:
            raise RuntimeError(
                f"RWKV-7 artifact metadata references a missing state tensor: {name}"
            ) from error
        if not isinstance(parameter, torch.nn.Parameter):
            raise RuntimeError(
                f"RWKV-7 protected state tensor is not a Parameter: {name}"
            )

    return {
        "quantized_linear_module_count": len(contract.vllm.quantized_modules),
        "protected_linear_module_count": len(contract.vllm.protected_linear_modules),
        "protected_embedding_module_count": len(
            contract.vllm.protected_embedding_modules
        ),
        "protected_normalization_module_count": len(
            contract.vllm.protected_normalization_modules
        ),
        "protected_state_tensor_count": len(contract.vllm.protected_state_tensors),
    }


def _snapshot_rwkv7_protected_parameters(
    model: torch.nn.Module,
    contract: RWKV7ArtifactContract,
) -> dict[str, dict[str, Any]]:
    """Capture exact protected values and ownership before quantization."""

    _validate_rwkv7_artifact_model_ownership(model, contract)
    snapshot = {}
    for name in contract.vllm.protected_parameter_keys:
        parameter = model.get_parameter(name)
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        snapshot[name] = {
            "owner_id": id(owner),
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "sha256": _tensor_sha256(parameter),
        }
    return snapshot


def _verify_rwkv7_protected_parameters(
    model: torch.nn.Module,
    contract: RWKV7ArtifactContract,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail if quantization changes protected ownership, shape, dtype, or value."""

    _validate_rwkv7_artifact_model_ownership(model, contract)
    expected_names = set(contract.vllm.protected_parameter_keys)
    if set(snapshot) != expected_names:
        raise RuntimeError(
            "RWKV-7 protected snapshot differs from the artifact inventory"
        )
    sha256 = {}
    for name in contract.vllm.protected_parameter_keys:
        expected = snapshot[name]
        parameter = model.get_parameter(name)
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        if id(owner) != expected.get("owner_id"):
            raise RuntimeError(
                f"RWKV-7 quantization replaced a protected module: {name}"
            )
        if list(parameter.shape) != expected.get("shape"):
            raise RuntimeError(
                f"RWKV-7 quantization changed a protected tensor shape: {name}"
            )
        if str(parameter.dtype) != expected.get("dtype"):
            raise RuntimeError(
                f"RWKV-7 quantization changed a protected tensor dtype: {name}"
            )
        actual_sha256 = _tensor_sha256(parameter)
        if actual_sha256 != expected.get("sha256"):
            raise RuntimeError(
                f"RWKV-7 quantization changed a protected tensor value: {name}"
            )
        sha256[name] = actual_sha256
    return {
        "passed": True,
        "parameter_count": len(sha256),
        "module_identity_preserved": True,
        "parameter_ownership_preserved": True,
        "parameter_values_preserved": True,
        "parameter_sha256": sha256,
    }


def _validate_loaded_rwkv7_artifact_ownership(
    model: torch.nn.Module,
    contract: RWKV7ArtifactContract,
    runtime_dtype: torch.dtype,
    expected_protected_parameter_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the public dequantized Transformers ownership boundary."""

    inventory = _validate_rwkv7_artifact_model_ownership(model, contract)
    quantized = [model.get_submodule(name) for name in contract.vllm.quantized_modules]
    protected = [model.get_submodule(name) for name in contract.vllm.protected_modules]
    protected_parameters = [
        model.get_parameter(name) for name in contract.vllm.protected_parameter_keys
    ]
    if not all(
        getattr(module, "quantization_scheme", None) is not None for module in quantized
    ):
        raise RuntimeError("RWKV-7 loaded target lacks its quantization scheme")
    if not all(
        isinstance(getattr(module, "weight", None), torch.nn.Parameter)
        and module.weight.is_floating_point()
        and module.weight.dtype == runtime_dtype
        for module in quantized
    ):
        raise RuntimeError(
            "RWKV-7 public dequantization did not restore standard float weights"
        )
    if not all(
        getattr(module, "quantization_scheme", None) is None for module in protected
    ):
        raise RuntimeError("RWKV-7 protected module acquired a quantization scheme")
    if not all(parameter.dtype == runtime_dtype for parameter in protected_parameters):
        raise RuntimeError("RWKV-7 protected parameter has the wrong runtime dtype")
    protected_parameter_sha256 = {
        name: _tensor_sha256(model.get_parameter(name))
        for name in contract.vllm.protected_parameter_keys
    }
    if expected_protected_parameter_sha256 is not None:
        if set(expected_protected_parameter_sha256) != set(
            contract.vllm.protected_parameter_keys
        ):
            raise RuntimeError(
                "RWKV-7 fresh-load protected digest inventory differs from metadata"
            )
        mismatched = sorted(
            name
            for name, expected in expected_protected_parameter_sha256.items()
            if protected_parameter_sha256[name] != expected
        )
        if mismatched:
            raise RuntimeError(
                f"RWKV-7 fresh load changed protected tensor values: {mismatched}"
            )
    return {
        **inventory,
        "runtime_float_weights_restored": True,
        "quantized_scheme_count": len(quantized),
        "protected_parameter_count": len(protected_parameters),
        "protected_parameter_values_verified": (
            expected_protected_parameter_sha256 is not None
        ),
        "vllm_metadata_validated": True,
    }


def audit_rwkv7_quantized_checkpoint(
    output_dir: Path,
    expected_targets: list[str],
    candidate: str,
    artifact_contract: RWKV7ArtifactContract | None = None,
    *,
    expected_protected_parameter_sha256: Mapping[str, str] | None = None,
    require_operator_runtime_provenance: bool = True,
) -> dict[str, Any]:
    """Verify compressed tensor storage, not merely serialized recipe metadata."""
    from safetensors import safe_open

    if not expected_targets or len(expected_targets) != len(set(expected_targets)):
        raise ValueError("RWKV-7 audit targets must be non-empty and unique")

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    _validate_native_rwkv7_config(config)
    quantization = config.get("quantization_config", {})
    loaded_contract = _load_rwkv7_artifact_contract(output_dir)
    if artifact_contract is not None and loaded_contract != artifact_contract:
        raise RuntimeError("RWKV-7 serialized artifact contract drifted")
    transformers_provenance = (
        validate_rwkv7_transformers_provenance(loaded_contract.repository)
        if require_operator_runtime_provenance
        else _validate_rwkv7_transformers_source_provenance(loaded_contract.repository)
    )
    if transformers_provenance != loaded_contract.runtime_provenance:
        raise RuntimeError(
            "RWKV-7 serialized runtime provenance differs from the active runtime"
        )
    if candidate != loaded_contract.candidate:
        raise RuntimeError(
            "RWKV-7 audit candidate differs from serialized artifact contract"
        )
    if expected_targets != loaded_contract.vllm.quantized_modules:
        raise RuntimeError(
            "RWKV-7 audit targets differ from the complete serialized inventory"
        )
    expected_format = (
        "pack-quantized"
        if candidate == "w8a16-low-rank-critical-high"
        else "nvfp4-pack-quantized"
    )
    if loaded_contract.vllm.quantization_format != expected_format:
        raise RuntimeError(
            "RWKV-7 serialized quantization format differs from candidate"
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
    if not isinstance(group, dict) or group.get("targets") != ["Linear"]:
        raise RuntimeError(
            "RWKV-7 serialized quantization config group must target exactly ['Linear']"
        )
    expected_group = _expected_quantization_group(candidate, serialized=True)
    if group != expected_group:
        raise RuntimeError(
            "RWKV-7 serialized quantization config group differs from the exact "
            f"candidate contract: expected={expected_group} actual={group}"
        )
    serialized_ignore = quantization.get("ignore")
    if not isinstance(serialized_ignore, list) or not serialized_ignore:
        raise RuntimeError(
            "RWKV-7 serialized quantization ignore must be a non-empty list"
        )
    if not all(isinstance(name, str) and bool(name) for name in serialized_ignore):
        raise RuntimeError(
            "RWKV-7 serialized quantization ignore entries must be non-empty "
            "Linear FQN strings"
        )
    if len(serialized_ignore) != len(set(serialized_ignore)):
        raise RuntimeError(
            "RWKV-7 serialized quantization ignore must not contain duplicates"
        )
    expected_ignore = set(loaded_contract.vllm.protected_linear_modules)
    actual_ignore = set(serialized_ignore)
    missing_ignore = sorted(expected_ignore - actual_ignore)
    unexpected_ignore = sorted(actual_ignore - expected_ignore)
    if missing_ignore or unexpected_ignore:
        raise RuntimeError(
            "RWKV-7 serialized protected Linear ignore inventory differs from "
            f"candidate metadata: missing={missing_ignore} "
            f"unexpected={unexpected_ignore}"
        )
    input_quantized = group.get("input_activations") is not None
    if input_quantized != (candidate == "nvfp4-w4a4"):
        raise RuntimeError(
            "RWKV-7 checkpoint activation quantization differs from candidate"
        )
    tensors: dict[str, tuple[list[int], str]] = {}
    protected_parameter_sha256 = {}
    for shard in sorted(output_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensors:
                    raise RuntimeError(
                        f"RWKV-7 artifact duplicates a tensor across shards: {name}"
                    )
                tensors[name] = (
                    handle.get_slice(name).get_shape(),
                    handle.get_slice(name).get_dtype(),
                )
                if (
                    expected_protected_parameter_sha256 is not None
                    and name in expected_protected_parameter_sha256
                ):
                    protected_parameter_sha256[name] = _tensor_sha256(
                        handle.get_tensor(name)
                    )
    expected_target_set = set(expected_targets)
    expected_tensor_owners = {
        ".weight_packed": expected_target_set,
        ".weight_scale": expected_target_set,
        ".weight_shape": (
            expected_target_set
            if candidate == "w8a16-low-rank-critical-high"
            else set()
        ),
        ".weight_global_scale": (
            set()
            if candidate == "w8a16-low-rank-critical-high"
            else expected_target_set
        ),
        ".input_global_scale": (
            expected_target_set if candidate == "nvfp4-w4a4" else set()
        ),
    }
    for suffix, expected_owners in expected_tensor_owners.items():
        actual_owners = _tensor_owners(tensors, suffix)
        if actual_owners != expected_owners:
            raise RuntimeError(
                "RWKV-7 physical compressed tensor inventory drifted: "
                f"suffix={suffix} expected={sorted(expected_owners)} "
                f"actual={sorted(actual_owners)}"
            )
    legacy_weight_aliases = sorted(
        {f"{target}.weight" for target in expected_targets} & tensors.keys()
    )
    if legacy_weight_aliases:
        raise RuntimeError(
            "RWKV-7 artifact contains legacy raw weight aliases: "
            f"{legacy_weight_aliases}"
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
            raise RuntimeError(f"RWKV-7 target has drifted packed dtype: {target}")
        expected_scale_dtype = (
            "F32" if candidate == "w8a16-low-rank-critical-high" else "F8_E4M3"
        )
        if tensors[f"{target}.weight_scale"][1] != expected_scale_dtype:
            raise RuntimeError(f"RWKV-7 target has drifted scale dtype: {target}")
        if (
            candidate == "w8a16-low-rank-critical-high"
            and tensors[f"{target}.weight_shape"][1] != "I64"
        ):
            raise RuntimeError(f"RWKV-7 target has drifted shape dtype: {target}")
    protected_names = set(loaded_contract.vllm.protected_parameter_keys)
    missing_protected = sorted(protected_names - tensors.keys())
    if missing_protected:
        raise RuntimeError(
            "RWKV-7 artifact is missing protected physical tensors: "
            f"{missing_protected}"
        )
    if expected_protected_parameter_sha256 is not None:
        if set(expected_protected_parameter_sha256) != protected_names:
            raise RuntimeError(
                "RWKV-7 protected digest inventory differs from artifact metadata"
            )
        mismatched_protected = sorted(
            name
            for name, expected_sha256 in expected_protected_parameter_sha256.items()
            if protected_parameter_sha256.get(name) != expected_sha256
        )
        if mismatched_protected:
            raise RuntimeError(
                "RWKV-7 serialized protected tensor values drifted: "
                f"{mismatched_protected}"
            )
    for name in loaded_contract.vllm.protected_modules:
        if any(key.startswith(f"{name}.weight_") for key in tensors):
            raise RuntimeError(f"RWKV-7 protected module was compressed: {name}")
    for name in loaded_contract.vllm.protected_tensors:
        if any(key.startswith(f"{name}_") for key in tensors):
            raise RuntimeError(f"RWKV-7 protected tensor was transformed: {name}")
    expected_tensor_keys = _expected_physical_tensor_keys(loaded_contract)
    actual_tensor_keys = set(tensors)
    missing_tensor_keys = sorted(expected_tensor_keys - actual_tensor_keys)
    unexpected_tensor_keys = sorted(actual_tensor_keys - expected_tensor_keys)
    if missing_tensor_keys or unexpected_tensor_keys:
        raise RuntimeError(
            "RWKV-7 physical safetensors key set differs from the exact artifact "
            f"contract: missing={missing_tensor_keys} "
            f"unexpected={unexpected_tensor_keys}"
        )
    return {
        "format": expected_format,
        "targets": expected_targets,
        "quantized_weight_names": [f"{name}.weight" for name in expected_targets],
        "protected_parameter_count": len(protected_names),
        "protected_parameter_values_verified": (
            expected_protected_parameter_sha256 is not None
        ),
        "tensor_count": len(tensors),
        "expected_tensor_count": len(expected_tensor_keys),
        "input_quantized": input_quantized,
        "artifact_contract_serialized": True,
        "transformers_provenance": transformers_provenance.model_dump(mode="json"),
        "operator_runtime_provenance_required": require_operator_runtime_provenance,
        "standard_linear_ownership": True,
        "legacy_weight_aliases": legacy_weight_aliases,
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
    RWKV7ArtifactContract,
    RWKV7RepositoryContract,
    _validate_loaded_rwkv7_artifact_ownership,
    _validate_native_rwkv7_config,
    _validate_native_rwkv7_runtime,
    _validate_rwkv7_loading_info,
    _validate_rwkv7_transformers_source_provenance,
    validate_rwkv7_transformers_provenance,
)

def require(condition, message):
    if not condition:
        raise RuntimeError(f'RWKV-7 fresh reload contract failed: {message}')

with open(f'{sys.argv[1]}/config.json', encoding='utf-8') as config_handle:
    serialized_config = json.load(config_handle)
_validate_native_rwkv7_config(serialized_config)
serialized_contract = serialized_config['rwkv7_quantization_metadata']
contract_model = RWKV7ArtifactContract.model_validate(serialized_contract)
contract = contract_model.model_dump(mode='json')
generate_seed = int(sys.argv[2])
prompt_ids = json.loads(sys.argv[3])
max_new_tokens = int(sys.argv[4])
warmup_runs = int(sys.argv[5])
timed_runs = int(sys.argv[6])
execution_mode = sys.argv[7] if len(sys.argv) > 7 else 'forward-generate'
runtime_device = sys.argv[8] if len(sys.argv) > 8 else 'cuda'
protected_parameter_sha256 = (
    json.loads(sys.argv[9]) if len(sys.argv) > 9 else None
)
require(
    execution_mode in ('load-only', 'forward-generate'),
    'execution mode must be load-only or forward-generate',
)
require(
    runtime_device in ('cpu', 'cuda') or (
        runtime_device.startswith('cuda:')
        and runtime_device.removeprefix('cuda:').isdigit()
    ),
    'runtime device must be cpu or cuda',
)
repository_contract = RWKV7RepositoryContract.model_validate(
    contract_model.repository
)
if execution_mode == 'load-only':
    active_provenance = _validate_rwkv7_transformers_source_provenance(
        repository_contract
    )
else:
    active_provenance = validate_rwkv7_transformers_provenance(
        repository_contract
    )
if active_provenance != contract_model.runtime_provenance:
    raise RuntimeError(
        'RWKV-7 serialized provenance differs from the active runtime scope'
    )
transformers_provenance = active_provenance.model_dump(mode='json')

from transformers import Rwkv7Config
from transformers.models.rwkv7 import Rwkv7ForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

config = Rwkv7Config.from_pretrained(sys.argv[1], trust_remote_code=False)
runtime_dtype = config.dtype
require(isinstance(runtime_dtype, torch.dtype), 'config dtype is not torch.dtype')
require(warmup_runs >= 1, 'warmup_runs must be at least one')
require(timed_runs >= 1, 'timed_runs must be at least one')
require(
    getattr(config, 'rwkv7_quantization_metadata', None) == contract,
    'Rwkv7Config metadata differs from the serialized artifact contract',
)
require(contract['schema_version'] == 3, 'artifact schema_version must be 3')
require(
    contract['candidate'] in (
        'nvfp4-w4a4',
        'nvfp4-w4a16',
        'nvfp4-w4a16-protection-ablation',
        'w8a16-low-rank-critical-high',
    ),
    'artifact candidate is outside the closed set',
)
require(
    contract['vllm']['architecture'] == 'Rwkv7ForCausalLM',
    'loader architecture is not Rwkv7ForCausalLM',
)
require(
    contract['vllm']['source_format'] == 'standard_hf',
    'source format is not standard_hf',
)
require(
    contract['vllm']['legacy_pth_direct_load'] is False,
    'legacy PTH direct load must remain disabled',
)
require(
    contract['vllm']['linear_weight_suffix'] == 'weight',
    'Linear weight suffix is not the standard weight name',
)
require(
    contract['vllm']['linear_weight_layout'] == 'out-in',
    'Linear weight layout is not out-in',
)
require(
    set(contract['vllm']['protected_v_first_linear_modules']).isdisjoint(
        contract['vllm']['quantized_modules']
    ),
    'protected v_first modules overlap the quantized inventory',
)

is_cuda = runtime_device.startswith('cuda')
cuda_device = torch.device(runtime_device) if is_cuda else None
if is_cuda:
    require(torch.cuda.is_available(), 'CUDA execution requested but unavailable')
    torch.cuda.reset_peak_memory_stats(cuda_device)
    torch.cuda.synchronize(cuda_device)
load_started = time.perf_counter()
model, loading_info = Rwkv7ForCausalLM.from_pretrained(
    sys.argv[1],
    config=config,
    device_map=runtime_device,
    dtype=runtime_dtype,
    quantization_config=CompressedTensorsConfig(dequantize=True),
    output_loading_info=True,
    trust_remote_code=False,
)
_validate_native_rwkv7_runtime(config, model)
_validate_rwkv7_loading_info(loading_info)
missing_quantized_weights = []
model = model.to(dtype=runtime_dtype).eval()
if is_cuda:
    torch.cuda.synchronize(cuda_device)
load_latency_ms = (time.perf_counter() - load_started) * 1000.0
load_peak_allocated_bytes = (
    torch.cuda.max_memory_allocated(cuda_device) if is_cuda else None
)
load_peak_reserved_bytes = (
    torch.cuda.max_memory_reserved(cuda_device) if is_cuda else None
)
model_resident_allocated_bytes = (
    torch.cuda.memory_allocated(cuda_device) if is_cuda else None
)
model_resident_reserved_bytes = (
    torch.cuda.memory_reserved(cuda_device) if is_cuda else None
)
ownership = _validate_loaded_rwkv7_artifact_ownership(
    model,
    contract_model,
    runtime_dtype,
    protected_parameter_sha256,
)
load_evidence = {
    'passed': True,
    'loader': 'Rwkv7ForCausalLM.from_pretrained',
    'config_class': type(config).__name__,
    'model_class': type(model).__name__,
    'trust_remote_code': False,
    'strict_loading_info': True,
    'missing_quantized_weights': missing_quantized_weights,
    'quantized_low_rank_module_count': len(
        contract['vllm']['quantized_low_rank_modules']
    ),
    'protected_v_first_linear_module_count': len(
        contract['vllm']['protected_v_first_linear_modules']
    ),
    **ownership,
}

if execution_mode == 'load-only':
    print(json.dumps({
        'dtype': str(runtime_dtype),
        'logits_dtype': None,
        'quantized_module_count': len(contract['vllm']['quantized_modules']),
        'protected_module_count': len(contract['vllm']['protected_modules']),
        'protected_state_tensor_count': len(contract['vllm']['protected_tensors']),
        'protected_parameter_count': len(
            contract['vllm']['protected_parameter_keys']
        ),
        'artifact_contract_validated': True,
        'transformers_provenance': transformers_provenance,
        'execution_mode': execution_mode,
        'standard_linear_load': load_evidence,
        'runtime_measurement': {
            'scope': 'fresh-process-transformers-direct-class-load',
            'canonical_performance_acceptance': False,
            'device': runtime_device,
            'load_latency_ms': load_latency_ms,
        },
        'standard_generate': {'passed': False, 'executed': False},
    }))
    raise SystemExit(0)

prompt = torch.tensor([prompt_ids], device=runtime_device)

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
    if is_cuda:
        torch.cuda.synchronize(cuda_device)
    generate_baseline_allocated_bytes = (
        torch.cuda.memory_allocated(cuda_device) if is_cuda else None
    )
    generate_baseline_reserved_bytes = (
        torch.cuda.memory_reserved(cuda_device) if is_cuda else None
    )
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    latencies_ms = []
    generated = None
    for _ in range(timed_runs):
        if is_cuda:
            torch.cuda.synchronize(cuda_device)
        generate_started = time.perf_counter()
        generated = generate_once()
        if is_cuda:
            torch.cuda.synchronize(cuda_device)
        latencies_ms.append((time.perf_counter() - generate_started) * 1000.0)

require(generated is not None, 'timed generation produced no output')
require(torch.isfinite(logits).all(), 'forward logits contain non-finite values')
require(
    generated.shape == (1, len(prompt_ids) + max_new_tokens),
    'generated token shape differs from the requested decode length',
)
require(
    generated[0, :len(prompt_ids)].tolist() == prompt_ids,
    'generated token prefix differs from the input prompt',
)
sorted_latencies_ms = sorted(latencies_ms)
latency_p90_index = max(0, math.ceil(0.9 * timed_runs) - 1)
elapsed_seconds = sum(latencies_ms) / 1000.0
print(json.dumps({
    'dtype': str(runtime_dtype),
    'logits_dtype': str(logits.dtype),
    'quantized_module_count': len(contract['vllm']['quantized_modules']),
    'protected_module_count': len(contract['vllm']['protected_modules']),
    'protected_state_tensor_count': len(contract['vllm']['protected_tensors']),
    'protected_parameter_count': len(contract['vllm']['protected_parameter_keys']),
    'artifact_contract_validated': True,
    'transformers_provenance': transformers_provenance,
    'execution_mode': execution_mode,
    'standard_linear_load': load_evidence,
    'runtime_measurement': {
        'scope': 'fresh-process-transformers-generate-diagnostic',
        'canonical_performance_acceptance': False,
        'device_name': (
            torch.cuda.get_device_name(cuda_device) if is_cuda else 'cpu'
        ),
        'device_capability': (
            list(torch.cuda.get_device_capability(cuda_device)) if is_cuda else None
        ),
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
            'peak_allocated_bytes': (
                torch.cuda.max_memory_allocated(cuda_device) if is_cuda else None
            ),
            'peak_reserved_bytes': (
                torch.cuda.max_memory_reserved(cuda_device) if is_cuda else None
            ),
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
    fresh_reload_mode: Literal["load-only", "forward-generate"] = "forward-generate",
    fresh_reload_device: str = "cuda",
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
    if fresh_reload_mode not in {"load-only", "forward-generate"}:
        raise ValueError("fresh reload mode must be load-only or forward-generate")
    valid_fresh_device = fresh_reload_device in {"cpu", "cuda"} or (
        fresh_reload_device.startswith("cuda:")
        and fresh_reload_device.removeprefix("cuda:").isdigit()
    )
    if not valid_fresh_device:
        raise ValueError("fresh reload device must be cpu, cuda, or cuda:N")
    execution_candidates = (
        candidates if forced_candidate is None else (forced_candidate,)
    )
    framework_versions = _installed_framework_versions()
    _validate_framework_versions(framework_versions)
    serialization_only = fresh_reload_mode == "load-only"
    runtime_provenance = (
        _validate_rwkv7_transformers_source_provenance()
        if serialization_only
        else validate_rwkv7_transformers_provenance()
    )
    failures = []
    for candidate in execution_candidates:
        model = model_factory()
        modifier = _build_rwkv7_quantization_recipe(
            model,
            candidate,
            framework_versions=framework_versions,
            runtime_provenance=runtime_provenance,
        )
        artifact_contract = build_rwkv7_artifact_contract(
            modifier.target_policy_metadata,
            candidate,
            checkpoint=checkpoint_contract,
            serialization_only_provenance=(
                runtime_provenance if serialization_only else None
            ),
        )
        protected_snapshot = _snapshot_rwkv7_protected_parameters(
            model,
            artifact_contract,
        )
        quantization_device = next(model.parameters()).device
        cuda_measured = quantization_device.type == "cuda"
        destination = output_dir / candidate
        destination.mkdir(parents=True, exist_ok=True)
        try:
            if cuda_measured:
                torch.cuda.synchronize(quantization_device)
                torch.cuda.reset_peak_memory_stats(quantization_device)
            quantization_started = time.perf_counter()
            result = oneshot(
                model=model,
                dataset=calibration_dataset if candidate == "nvfp4-w4a4" else None,
                processor=processor if candidate == "nvfp4-w4a4" else None,
                recipe=modifier,
                pipeline="basic" if candidate == "nvfp4-w4a4" else "datafree",
                output_dir=None,
            )
            protection_audit = _verify_rwkv7_protected_parameters(
                result,
                artifact_contract,
                protected_snapshot,
            )
            if cuda_measured:
                torch.cuda.synchronize(quantization_device)
            quantization_runtime = {
                "scope": "llmcompressor-oneshot",
                "latency_ms": (time.perf_counter() - quantization_started) * 1000.0,
                "cuda_measured": cuda_measured,
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(quantization_device)
                    if cuda_measured
                    else None
                ),
                "peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved(quantization_device)
                    if cuda_measured
                    else None
                ),
            }
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
            if processor is not None and hasattr(processor, "save_pretrained"):
                processor.save_pretrained(destination)
            audit = audit_rwkv7_quantized_checkpoint(
                destination,
                modifier.target_policy_metadata.selection.names,
                candidate,
                artifact_contract,
                expected_protected_parameter_sha256=(
                    protection_audit["parameter_sha256"]
                ),
                require_operator_runtime_provenance=not serialization_only,
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
                fresh_reload_mode,
                fresh_reload_device,
                json.dumps(protection_audit["parameter_sha256"], sort_keys=True),
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
            "protection_audit": protection_audit,
            "cell_forward": {
                "passed": True,
                "output_shape": list(cell_output.shape),
                "state_shape": list(cell_state.shape),
            },
            "quantization_runtime": quantization_runtime,
            "provenance_scope": (
                "serialization-only"
                if serialization_only
                else "operator-runtime-validated"
            ),
            "formal_runtime_generation": (
                checkpoint_contract is not None
                and fresh_reload_mode == "forward-generate"
            ),
            "standard_linear_ownership": {
                "passed": True,
                "quantized_weight_names": (
                    artifact_contract.vllm.quantized_weight_names
                ),
                "quantized_low_rank_modules": (
                    artifact_contract.vllm.quantized_low_rank_modules
                ),
                "protected_v_first_linear_modules": (
                    artifact_contract.vllm.protected_v_first_linear_modules
                ),
            },
            "fresh_reload": {
                "passed": reload_run.returncode == 0,
                "returncode": reload_run.returncode,
                "mode": fresh_reload_mode,
                "device": fresh_reload_device,
                "stdout": reload_run.stdout[-8000:],
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
        if reload_run.returncode != 0:
            failures.append(
                {
                    "candidate": candidate,
                    "stage": (
                        "fresh_reload_load"
                        if fresh_reload_mode == "load-only"
                        else "fresh_reload_generate"
                    ),
                    "error_type": "SubprocessError",
                    "error": reload_run.stderr[-8000:],
                }
            )
            _atomic_json(destination / "rwkv7_quantization_execution.json", metadata)
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


def _artifact_file_manifest(
    directory: Path,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    excluded = set() if exclude is None else exclude
    files = []
    for path in sorted(
        candidate for candidate in directory.rglob("*") if candidate.is_file()
    ):
        relative_path = path.relative_to(directory).as_posix()
        if relative_path in excluded:
            continue
        files.append(
            {
                "path": relative_path,
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


def _rwkv7_tokenizer_provenance(tokenizer_path: Path) -> dict[str, Any]:
    tokenizer_path = tokenizer_path.resolve()
    if not tokenizer_path.is_dir() or not (tokenizer_path / "tokenizer.json").is_file():
        raise ValueError(
            "RWKV-7 formal candidate runner requires a local standard fast "
            "tokenizer directory containing tokenizer.json"
        )
    return {
        "source_path": str(tokenizer_path),
        "artifact_manifest": _artifact_file_manifest(tokenizer_path),
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
    tokenizer_path: Path,
    destination: Path,
    checkpoint_contract: RWKV7CheckpointContract,
    runtime_provenance: RWKV7TransformersProvenance,
    implementation_provenance: RWKV7ImplementationProvenance,
) -> dict[str, Any]:
    provenance_path = destination / "rwkv7_source_provenance.json"
    provenance_name = provenance_path.name
    tokenizer_provenance = _rwkv7_tokenizer_provenance(tokenizer_path)
    provenance_prefix = {
        "schema_version": 2,
        "checkpoint": checkpoint_contract.model_dump(mode="json"),
        "tokenizer": tokenizer_provenance,
        "converter_runtime": runtime_provenance.model_dump(mode="json"),
        "implementation": implementation_provenance.model_dump(mode="json"),
    }
    if destination.exists() and any(destination.iterdir()):
        if not provenance_path.is_file():
            raise RuntimeError(
                "standard checkpoint destination is non-empty without provenance"
            )
        actual_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_provenance = {
            **provenance_prefix,
            "artifact_manifest": _artifact_file_manifest(
                destination,
                exclude={provenance_name},
            ),
        }
        if actual_provenance != expected_provenance:
            raise RuntimeError(
                "standard checkpoint converter provenance or manifest drifted"
            )
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
            tokenizer_name_or_path=str(tokenizer_path.resolve()),
            source_revision=checkpoint_contract.revision,
        )
        expected_provenance = {
            **provenance_prefix,
            "artifact_manifest": _artifact_file_manifest(destination),
        }
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
    tokenizer_path: Path,
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
    fresh_reload_mode: Literal["load-only", "forward-generate"] = ("forward-generate"),
) -> dict[str, Any]:
    """Quantize the pinned 1.5B checkpoint and emit a traceable candidate artifact."""

    if candidate not in _CANDIDATE_SCHEMES:
        raise ValueError(f"unsupported RWKV-7 candidate: {candidate}")
    if fresh_reload_mode not in {"load-only", "forward-generate"}:
        raise ValueError(
            "formal RWKV-7 fresh reload mode must be load-only or forward-generate"
        )
    serialization_only = fresh_reload_mode == "load-only"
    runtime_provenance = (
        _validate_rwkv7_transformers_source_provenance()
        if serialization_only
        else validate_rwkv7_transformers_provenance()
    )
    implementation_provenance = validate_rwkv7_implementation_provenance(
        implementation_revision
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12:
        raise RuntimeError("formal RWKV-7 NVFP4 execution requires a Blackwell GPU")

    checkpoint_contract = verify_rwkv7_checkpoint(checkpoint_path)
    tokenizer_provenance = _rwkv7_tokenizer_provenance(tokenizer_path)
    output_dir = output_dir.resolve()
    standard_checkpoint = output_dir / "baseline-standard-hf"
    standard_manifest = _prepare_standard_rwkv7_checkpoint(
        checkpoint_path.resolve(),
        tokenizer_path.resolve(),
        standard_checkpoint,
        checkpoint_contract,
        runtime_provenance,
        implementation_provenance,
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
        fresh_reload_mode=fresh_reload_mode,
    )
    candidate_dir = output_dir / "candidates" / candidate
    artifact_runtime_provenance = execution["artifact_contract"]["runtime_provenance"]
    if artifact_runtime_provenance != runtime_provenance.model_dump(mode="json"):
        raise RuntimeError(
            "RWKV-7 runtime provenance drifted during formal candidate execution"
        )
    result = {
        "schema_version": 1,
        "implementation_revision": implementation_revision,
        "implementation": implementation_provenance.model_dump(mode="json"),
        "repository": RWKV7RepositoryContract().model_dump(mode="json"),
        "runtime_provenance": runtime_provenance.model_dump(mode="json"),
        "runtime_provenance_scope": (
            "serialization-only" if serialization_only else "operator-runtime-validated"
        ),
        "checkpoint": checkpoint_contract.model_dump(mode="json"),
        "tokenizer": tokenizer_provenance,
        "standard_checkpoint": standard_manifest,
        "calibration": calibration,
        "candidate": candidate,
        "execution": execution,
        "candidate_artifact": _artifact_file_manifest(
            candidate_dir,
            exclude={"rwkv7_candidate_result.json"},
        ),
        "formal_checkpoint": True,
        "formal_runtime_generation": not serialization_only,
        "formal_evaluation": False,
        "diagnostic_tiny": False,
    }
    _atomic_json(candidate_dir / "rwkv7_candidate_result.json", result)
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


def _attention_module_ignore(
    base_model_prefix: str,
    module_names: tuple[str, ...],
) -> str:
    alternatives = "|".join(re.escape(name) for name in module_names)
    return (
        rf"re:^{re.escape(base_model_prefix)}\.blocks\.\d+\.att\."
        rf"({alternatives})$"
    )


def _attention_ignore(base_model_prefix: str) -> str:
    return _attention_module_ignore(
        base_model_prefix,
        (
            *_TIME_MIX_LINEAR_NAMES,
            *_LOW_RANK_LINEAR_NAMES,
            *_V_FIRST_LINEAR_NAMES,
        ),
    )


def _attention_value_ignore(base_model_prefix: str) -> str:
    return _attention_module_ignore(
        base_model_prefix,
        ("value", *_V_FIRST_LINEAR_NAMES),
    )


def _low_rank_w8_attention_ignore(base_model_prefix: str) -> str:
    return _attention_module_ignore(
        base_model_prefix,
        (*_TIME_MIX_LINEAR_NAMES, *_V_FIRST_LINEAR_NAMES),
    )


def _vllm_nvfp4_protection_ablation_ignore(
    base_model_prefix: str,
) -> list[str]:
    escaped_prefix = re.escape(base_model_prefix)
    return [
        rf"re:^{escaped_prefix}\.blocks\.0\.att\.value$",
        rf"re:^{escaped_prefix}\.blocks\.\d+\.ffn\.(?:key|value)$",
        _HEAD_IGNORE,
    ]


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
    """Validate standard RWKV-7 structure and select candidate-owned Linears.

    The validation is intentionally completed before the quantization config is
    applied. Any architecture drift therefore fails without partially modifying
    the model.
    """

    base_model_prefix, base_model = _resolve_standard_base_model(model)
    attention_ignore = _attention_ignore(base_model_prefix)
    low_rank_w8_attention_ignore = _low_rank_w8_attention_ignore(base_model_prefix)
    if protection_profile == "critical-high":
        required_ignore = [attention_ignore, _HEAD_IGNORE]
    elif protection_profile == "low-rank-w8-critical-high":
        required_ignore = [low_rank_w8_attention_ignore, _HEAD_IGNORE]
    elif protection_profile == "v-first-dataflow":
        required_ignore = _vllm_nvfp4_protection_ablation_ignore(base_model_prefix)
    else:
        raise ValueError(f"unsupported RWKV-7 protection profile: {protection_profile}")
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

    first_value_module: list[str] = []
    later_value_modules: list[str] = []
    later_v_first_linear_modules: list[str] = []
    other_time_mix_modules: list[str] = []
    later_v_first_tensors: list[str] = []
    low_rank_modules: list[str] = []
    channel_mix_modules: list[str] = []
    recurrent_state_tensors: list[str] = []
    embedding_modules: list[str] = []
    normalization_modules: list[str] = []
    expected_linear_modules = {_HEAD_IGNORE}

    _require_module(model, "head", torch.nn.Linear, "head")
    embeddings_path = f"{base_model_prefix}.embeddings"
    _require_module(base_model, "embeddings", torch.nn.Embedding, embeddings_path)
    embedding_modules.append(embeddings_path)
    output_norm_path = f"{base_model_prefix}.ln_out"
    _require_module(base_model, "ln_out", torch.nn.LayerNorm, output_norm_path)
    normalization_modules.append(output_norm_path)
    for layer_id, block in enumerate(blocks):
        block_path = f"{base_model_prefix}.blocks.{layer_id}"

        if layer_id == 0:
            input_norm_type = (
                torch.nn.Identity
                if getattr(config, "embedding_layer_norm_fused", False)
                else torch.nn.LayerNorm
            )
            _require_module(block, "ln0", input_norm_type, f"{block_path}.ln0")
            if input_norm_type is torch.nn.LayerNorm:
                normalization_modules.append(f"{block_path}.ln0")
        else:
            _require_module(
                block,
                "ln0",
                torch.nn.Identity,
                f"{block_path}.ln0",
            )
        for norm_name in ("ln1", "ln2"):
            norm_path = f"{block_path}.{norm_name}"
            _require_module(block, norm_name, torch.nn.LayerNorm, norm_path)
            normalization_modules.append(norm_path)

        attention = _require_module(block, "att", torch.nn.Module, f"{block_path}.att")
        channel_mix = _require_module(
            block, "ffn", torch.nn.Module, f"{block_path}.ffn"
        )
        if getattr(attention, "layer_id", None) != layer_id:
            raise ValueError(
                f"RWKV-7 target policy requires `{block_path}.att.layer_id == "
                f"{layer_id}`."
            )
        attention_norm_path = f"{block_path}.att.ln_x"
        _require_module(
            attention,
            "ln_x",
            torch.nn.GroupNorm,
            attention_norm_path,
        )
        normalization_modules.append(attention_norm_path)

        for parameter_name in _TIME_MIX_PARAMETER_NAMES:
            parameter_path = f"{block_path}.att.{parameter_name}"
            _require_parameter(
                attention,
                parameter_name,
                parameter_path,
            )
            recurrent_state_tensors.append(parameter_path)

        for linear_name in _LOW_RANK_LINEAR_NAMES:
            module_path = f"{block_path}.att.{linear_name}"
            _require_module(attention, linear_name, torch.nn.Linear, module_path)
            low_rank_modules.append(module_path)
            expected_linear_modules.add(module_path)

        v0_path = f"{block_path}.att.v0"
        if layer_id == 0:
            for name in ("v0", *_V_FIRST_LINEAR_NAMES):
                if hasattr(attention, name):
                    raise ValueError(
                        "RWKV-7 target policy requires layer 0 to produce "
                        f"`v_first` without `{block_path}.att.{name}`."
                    )
        else:
            _require_parameter(attention, "v0", v0_path)
            later_v_first_tensors.append(v0_path)
            for linear_name in _V_FIRST_LINEAR_NAMES:
                module_path = f"{block_path}.att.{linear_name}"
                _require_module(
                    attention,
                    linear_name,
                    torch.nn.Linear,
                    module_path,
                )
                later_v_first_linear_modules.append(module_path)
                expected_linear_modules.add(module_path)

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

        channel_state_path = f"{block_path}.ffn.x_k"
        _require_parameter(channel_mix, "x_k", channel_state_path)
        recurrent_state_tensors.append(channel_state_path)
        for linear_name in ("key", "value"):
            module_path = f"{block_path}.ffn.{linear_name}"
            _require_module(channel_mix, linear_name, torch.nn.Linear, module_path)
            channel_mix_modules.append(module_path)
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

    policy_ignore = list(dict.fromkeys([*ignore, *required_ignore]))
    if protection_profile == "critical-high":
        selected_modules = channel_mix_modules
    elif protection_profile == "low-rank-w8-critical-high":
        selected_modules = [*low_rank_modules, *channel_mix_modules]
    else:
        selected_modules = _vllm_nvfp4_protection_ablation_targets(
            base_model_prefix,
            len(blocks),
        )
        missing_ablation_targets = sorted(
            set(selected_modules) - expected_linear_modules
        )
        if missing_ablation_targets:
            raise ValueError(
                "RWKV-7 vLLM NVFP4 protection ablation requires the exact "
                "no-FFN 9+12*(L-1) consumer matrix; "
                f"missing={missing_ablation_targets}"
            )

    selected_module_set = set(selected_modules)

    def protected_decision(
        names: list[str],
        reason: str,
    ) -> QuantizationTargetPolicyDecision | None:
        protected_names = [name for name in names if name not in selected_module_set]
        if not protected_names:
            return None
        return QuantizationTargetPolicyDecision(
            kind="module",
            names=protected_names,
            reason=reason,
        )

    dataflow_module_protections = [
        protected_decision(
            first_value_module,
            "Layer-0 TimeMix value projection produces v_first, which is "
            "consumed by every later RWKV-7 block.",
        ),
        protected_decision(
            later_value_modules,
            "Later TimeMix value projections outside the declared ablation "
            "matrix remain high precision.",
        ),
        protected_decision(
            later_v_first_linear_modules,
            "The v1/v2 Linear modules outside the declared ablation matrix "
            "remain on the protected v_first path.",
        ),
    ]
    remaining_linear_protections = [
        protected_decision(
            other_time_mix_modules,
            "TimeMix projections outside the candidate-owned target matrix "
            "remain high precision.",
        ),
        protected_decision(
            low_rank_modules,
            "Low-rank Linear modules outside the candidate-owned target matrix "
            "remain high precision.",
        ),
        protected_decision(
            channel_mix_modules,
            "ChannelMix projections outside the candidate-owned target matrix "
            "remain high precision.",
        ),
    ]
    protections = [
        decision for decision in dataflow_module_protections if decision is not None
    ]
    protections.extend(
        [
            QuantizationTargetPolicyDecision(
                kind="tensor",
                names=later_v_first_tensors,
                reason=(
                    "The v0 tensor gates every later layer's dependency on v_first "
                    "and remains high precision in every candidate."
                ),
            )
        ]
    )
    protections.extend(
        decision for decision in remaining_linear_protections if decision is not None
    )
    protections.extend(
        [
            QuantizationTargetPolicyDecision(
                kind="module",
                names=embedding_modules,
                reason=(
                    "Keep token embeddings high precision outside every Linear "
                    "quantization candidate."
                ),
            ),
            QuantizationTargetPolicyDecision(
                kind="module",
                names=normalization_modules,
                reason=(
                    "Keep every LayerNorm and GroupNorm high precision outside "
                    "every Linear quantization candidate."
                ),
            ),
        ]
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
                names=recurrent_state_tensors,
                reason=(
                    "Keep all recurrent TimeMix and ChannelMix state Parameters "
                    "high precision; v0 remains on the protected v_first path."
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
                "Select standard RWKV-7 Linear modules resolved by the candidate: "
                "ChannelMix for critical-high candidates, w/a/g low-rank modules "
                "for W8, or the exact no-FFN 9+12*(L-1) vLLM NVFP4 consumer "
                "matrix for the protection ablation."
            ),
        ),
        protections=protections,
    )
    return policy_ignore, metadata
