"""Conservative quantization targeting for standard Transformers RWKV-7."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "QuantizationTargetPolicyDecision",
    "QuantizationTargetPolicyMetadata",
    "RWKV7QuantizationRecipeMetadata",
    "apply_rwkv7_target_policy",
    "build_rwkv7_quantization_recipe",
    "quantize_rwkv7_oneshot",
    "audit_rwkv7_quantized_checkpoint",
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
_SUPPORTED_FRAMEWORK_VERSIONS = {
    "compressed_tensors": "0.17.2.a20260731",
    "transformers": "5.15.0.dev0",
}
_CANDIDATE_SCHEMES = {
    "nvfp4-w4a4": "NVFP4",
    "nvfp4-w4a16": "NVFP4A16",
}
_FRESH_RELOAD_GENERATE_SEED = 20260801
_FRESH_RELOAD_PROMPT_IDS = [1, 2, 3, 4]
_FRESH_RELOAD_NEW_TOKENS = 4


class RWKV7QuantizationRecipeMetadata(BaseModel):
    """Loader-facing contract for one closed RWKV-7 quantization candidate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    candidate: Literal["nvfp4-w4a4", "nvfp4-w4a16"]
    candidate_order: list[str]
    algorithm: Literal["NVFP4"] = "NVFP4"
    weight_dtype: Literal["float4"] = "float4"
    weight_group_size: Literal[16] = 16
    weight_scale_dtype: Literal["float8_e4m3fn"] = "float8_e4m3fn"
    input_dtype: Literal["float4", "float16"]
    input_scale: Literal["dynamic_local", "none"]
    input_scale_dtype: Literal["float8_e4m3fn"] | None
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
    base_model_prefix: str
    selection: QuantizationTargetPolicyDecision
    protections: list[QuantizationTargetPolicyDecision]
    recipe: RWKV7QuantizationRecipeMetadata | None = None


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
            "RWKV-7 NVFP4 recipe requires the validated framework versions: "
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
        input_dtype="float4" if inputs is not None else "float16",
        input_scale="dynamic_local" if inputs is not None else "none",
        input_scale_dtype="float8_e4m3fn" if inputs is not None else None,
        targets=targets,
        framework_versions=framework_versions,
    )


def build_rwkv7_quantization_recipe(
    model: torch.nn.Module,
    candidate: str = "nvfp4-w4a4",
    *,
    framework_versions: dict[str, str] | None = None,
):
    """Build a validated NVFP4-first recipe without applying quantization."""

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
    )
    modifier._apply_target_policy(model)
    recipe_metadata = _validate_candidate_scheme(
        candidate,
        next(iter(modifier.resolved_config.config_groups.values())),
        targets=list(modifier.target_policy_metadata.selection.names),
        framework_versions=versions,
    )
    modifier.target_policy_metadata = modifier.target_policy_metadata.model_copy(
        update={"recipe": recipe_metadata}
    )
    return modifier


def audit_rwkv7_quantized_checkpoint(
    output_dir: Path, expected_targets: list[str], candidate: str
) -> dict[str, Any]:
    """Verify compressed tensor storage, not merely serialized recipe metadata."""
    from safetensors import safe_open

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config", {})
    expected_format = "nvfp4-pack-quantized"
    if (
        quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("quantization_status") != "compressed"
        or quantization.get("format") != expected_format
    ):
        raise RuntimeError("RWKV-7 checkpoint lacks compressed NVFP4 metadata")
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
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensors[name] = (
                    handle.get_slice(name).get_shape(),
                    handle.get_slice(name).get_dtype(),
                )
    for target in expected_targets:
        required = {
            f"{target}.weight_packed",
            f"{target}.weight_scale",
            f"{target}.weight_global_scale",
        }
        if candidate == "nvfp4-w4a4":
            required.add(f"{target}.input_global_scale")
        elif f"{target}.input_global_scale" in tensors:
            raise RuntimeError(
                f"W4A16 target unexpectedly quantized its input: {target}"
            )
        missing = sorted(required - tensors.keys())
        if missing or f"{target}.weight" in tensors:
            raise RuntimeError(
                "RWKV-7 target was not physically NVFP4-compressed: "
                f"{target}; missing={missing}"
            )
        if (
            tensors[f"{target}.weight_packed"][1] != "U8"
            or tensors[f"{target}.weight_scale"][1] != "F8_E4M3"
        ):
            raise RuntimeError(
                f"RWKV-7 target has drifted packed/scale dtypes: {target}"
            )
    protected = [name for name in tensors if (".att." in name or name == "head.weight")]
    if any(
        name.endswith(("weight_packed", "weight_scale", "weight_global_scale"))
        for name in protected
    ):
        raise RuntimeError("RWKV-7 protected TimeMix/head tensors were compressed")
    return {
        "format": expected_format,
        "targets": expected_targets,
        "protected_tensor_count": len(protected),
        "tensor_count": len(tensors),
        "input_quantized": input_quantized,
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


def quantize_rwkv7_oneshot(
    model_factory: Callable[[], torch.nn.Module],
    output_dir: Path,
    *,
    calibration_dataset: object | None,
    processor: object | None,
    candidates: tuple[str, ...] = ("nvfp4-w4a4", "nvfp4-w4a16"),
    forced_candidate: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Execute the closed candidate order through standard ``oneshot``."""
    from llmcompressor import oneshot

    if candidates != tuple(_CANDIDATE_SCHEMES):
        raise ValueError(
            "RWKV-7 quantization execution requires the closed candidate order"
        )
    if forced_candidate is not None and forced_candidate not in _CANDIDATE_SCHEMES:
        raise ValueError(f"unsupported forced RWKV-7 candidate: {forced_candidate}")
    execution_candidates = (
        candidates if forced_candidate is None else (forced_candidate,)
    )
    failures = []
    for candidate in execution_candidates:
        model = model_factory()
        modifier = build_rwkv7_quantization_recipe(model, candidate)
        destination = output_dir / candidate
        destination.mkdir(parents=True, exist_ok=True)
        try:
            result = oneshot(
                model=model,
                dataset=calibration_dataset if candidate == "nvfp4-w4a4" else None,
                processor=processor if candidate == "nvfp4-w4a4" else None,
                recipe=modifier,
                pipeline="basic" if candidate == "nvfp4-w4a4" else "datafree",
                output_dir=None,
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
            result.save_pretrained(destination, save_compressed=True)
            if processor is not None and hasattr(processor, "save_pretrained"):
                processor.save_pretrained(destination)
            audit = audit_rwkv7_quantized_checkpoint(
                destination, modifier.target_policy_metadata.selection.names, candidate
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
        reload_script = """
import json, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

config = AutoConfig.from_pretrained(sys.argv[1])
generate_seed = int(sys.argv[2])
prompt_ids = json.loads(sys.argv[3])
max_new_tokens = int(sys.argv[4])
runtime_dtype = config.dtype
assert isinstance(runtime_dtype, torch.dtype)
model = AutoModelForCausalLM.from_pretrained(
    sys.argv[1],
    device_map='cuda',
    dtype=runtime_dtype,
    quantization_config=CompressedTensorsConfig(dequantize=True),
).to(dtype=runtime_dtype).eval()
quantized = [
    module
    for block in model.model.blocks
    for module in (block.ffn.key, block.ffn.value)
]
protected = [
    model.head,
    *[
        module
        for block in model.model.blocks
        for module in (
            block.att.receptance,
            block.att.key,
            block.att.value,
            block.att.output,
        )
    ],
]
assert all(
    getattr(module, 'quantization_scheme', None) is not None for module in quantized
)
assert all(module.weight.dtype == runtime_dtype for module in quantized)
assert all(getattr(module, 'quantization_scheme', None) is None for module in protected)
assert all(module.weight.dtype == runtime_dtype for module in protected)
prompt = torch.tensor([prompt_ids], device='cuda')
torch.manual_seed(generate_seed)
with torch.inference_mode():
    logits = model(prompt).logits
    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_k=8,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=[],
    )
assert torch.isfinite(logits).all()
assert generated.shape == (1, len(prompt_ids) + max_new_tokens)
assert generated[0, :len(prompt_ids)].tolist() == prompt_ids
print(json.dumps({
    'dtype': str(runtime_dtype),
    'logits_dtype': str(logits.dtype),
    'quantized_module_count': len(quantized),
    'protected_module_count': len(protected),
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
        reload_environment = dict(os.environ)
        reload_temporary = destination / ".fresh-reload-tmp"
        reload_temporary.mkdir(exist_ok=True)
        reload_environment["TMPDIR"] = str(reload_temporary)
        reload_run = subprocess.run(
            [
                sys.executable,
                "-c",
                reload_script,
                str(destination),
                str(_FRESH_RELOAD_GENERATE_SEED),
                json.dumps(_FRESH_RELOAD_PROMPT_IDS),
                str(_FRESH_RELOAD_NEW_TOKENS),
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
            "audit": audit,
            "cell_forward": {
                "passed": True,
                "output_shape": list(cell_output.shape),
                "state_shape": list(cell_state.shape),
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
        return result, metadata
    raise RuntimeError(f"all RWKV-7 quantization candidates failed: {failures}")


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
    attention_ignore: str,
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

    allowed_ignore = {attention_ignore, _HEAD_IGNORE}
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
) -> tuple[list[str], QuantizationTargetPolicyMetadata]:
    """Validate standard RWKV-7 structure and select only ChannelMix linears.

    The validation is intentionally completed before the quantization config is
    applied. Any architecture drift therefore fails without partially modifying
    the model.
    """

    base_model_prefix, base_model = _resolve_standard_base_model(model)
    attention_ignore = _attention_ignore(base_model_prefix)
    _validate_policy_inputs(
        resolved_targets,
        ignore,
        kv_cache_enabled,
        attention_ignore,
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

    policy_ignore = list(dict.fromkeys([*ignore, attention_ignore, _HEAD_IGNORE]))
    metadata = QuantizationTargetPolicyMetadata(
        base_model_prefix=base_model_prefix,
        selection=QuantizationTargetPolicyDecision(
            kind="module",
            names=selected_modules,
            reason=(
                "Select only standard RWKV-7 ChannelMix key/value projections; "
                "they are outside the recurrent TimeMix and v_first dataflow."
            ),
        ),
        protections=[
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
                    "Later TimeMix value projections are blended with v_first "
                    "before entering recurrent WKV state updates."
                ),
            ),
            QuantizationTargetPolicyDecision(
                kind="tensor",
                names=later_value_tensors,
                reason=(
                    "The v0/v1/v2 tensors gate every later layer's dependency on "
                    "v_first and are not Linear module targets."
                ),
            ),
            QuantizationTargetPolicyDecision(
                kind="module",
                names=other_time_mix_modules,
                reason=(
                    "Keep the remaining TimeMix projections out of this minimal "
                    "policy because they feed recurrent WKV and local attention."
                ),
            ),
            QuantizationTargetPolicyDecision(
                kind="module",
                names=[_HEAD_IGNORE],
                reason=(
                    "Keep the output head unquantized so this initial policy only "
                    "changes repeated ChannelMix projections."
                ),
            ),
            QuantizationTargetPolicyDecision(
                kind="tensor",
                names=recurrent_time_mix_tensors,
                reason=(
                    "Protect the standard TimeMix parameters that control recurrent "
                    "WKV state, receptance, decay, key, value, and gating."
                ),
            ),
        ],
    )
    return policy_ignore, metadata
