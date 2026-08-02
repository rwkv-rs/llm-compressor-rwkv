"""Conservative quantization targeting for standard Transformers RWKV-7."""

import re
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "QuantizationTargetPolicyDecision",
    "QuantizationTargetPolicyMetadata",
    "RWKV7QuantizationRecipeMetadata",
    "apply_rwkv7_target_policy",
    "build_rwkv7_quantization_recipe",
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
