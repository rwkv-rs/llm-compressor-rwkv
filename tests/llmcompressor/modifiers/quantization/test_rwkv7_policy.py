"""CPU tests for the fail-closed RWKV-7 quantization target policy."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.utils import match_named_modules

from llmcompressor.core import State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.rwkv7 import (
    QuantizationTargetPolicyMetadata,
    RWKV7TransformersProvenance,
    apply_rwkv7_target_policy,
    build_rwkv7_quantization_recipe,
)
from llmcompressor.recipe import Recipe


class _Rwkv7Config:
    model_type = "rwkv7"

    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers


class _TimeMix(torch.nn.Module):
    def __init__(self, layer_id: int, hidden_size: int):
        super().__init__()
        self.layer_id = layer_id
        for name in (
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
        ):
            setattr(self, name, torch.nn.Parameter(torch.zeros(hidden_size)))
        for name in ("w1", "w2", "a1", "a2", "g1", "g2"):
            setattr(
                self,
                name,
                torch.nn.Linear(hidden_size, hidden_size, bias=False),
            )
        if layer_id > 0:
            self.v0 = torch.nn.Parameter(torch.zeros(hidden_size))
            self.v1 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
            self.v2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_x = torch.nn.GroupNorm(1, hidden_size)


class _ChannelMix(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.x_k = torch.nn.Parameter(torch.zeros(hidden_size))
        self.key = torch.nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.value = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)


class _Block(torch.nn.Module):
    def __init__(self, layer_id: int, hidden_size: int):
        super().__init__()
        if layer_id == 0:
            self.ln0 = torch.nn.LayerNorm(hidden_size)
        else:
            self.ln0 = torch.nn.Identity()
        self.ln1 = torch.nn.LayerNorm(hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size)
        self.att = _TimeMix(layer_id, hidden_size)
        self.ffn = _ChannelMix(hidden_size)


class _Rwkv7Model(torch.nn.Module):
    def __init__(self, config: _Rwkv7Config, hidden_size: int):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(hidden_size, hidden_size)
        self.blocks = torch.nn.ModuleList(
            [
                _Block(layer_id, hidden_size)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_out = torch.nn.LayerNorm(hidden_size)


class _Rwkv7ForCausalLM(torch.nn.Module):
    base_model_prefix = "model"

    def __init__(self, num_hidden_layers: int = 2, hidden_size: int = 8):
        super().__init__()
        self.config = _Rwkv7Config(num_hidden_layers)
        self.model = _Rwkv7Model(self.config, hidden_size)
        self.head = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)


@pytest.fixture
def real_rwkv7_types():
    """Select the installed Transformers classes for tiny integration cases."""


@pytest.fixture(autouse=True)
def _standard_rwkv7_types(monkeypatch, request):
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7."
        "validate_rwkv7_transformers_provenance",
        lambda *_args: RWKV7TransformersProvenance(
            repository="https://github.com/rwkv-rs/transformers-rwkv.git",
            revision="5d11fbe2559fec5611798bd6cc3f6c89ae145f68",
            installation_source="editable-git",
            editable=True,
        ),
    )
    if "real_rwkv7_types" in request.fixturenames:
        return
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7._get_rwkv7_model_types",
        lambda: (_Rwkv7Config, _Rwkv7ForCausalLM),
    )


def _apply_policy(model: torch.nn.Module):
    return apply_rwkv7_target_policy(
        model=model,
        resolved_targets={"Linear"},
        ignore=[],
        kv_cache_enabled=False,
    )


@pytest.mark.unit
def test_rwkv7_policy_selects_channel_mix_and_records_recurrent_protections():
    model = _Rwkv7ForCausalLM()

    ignore, metadata = _apply_policy(model)

    assert metadata.base_model_prefix == "model"
    assert ignore == [
        (
            r"re:^model\.blocks\.\d+\.att\."
            r"(receptance|key|value|output|w1|w2|a1|a2|g1|g2|v1|v2)$"
        ),
        "head",
    ]
    assert metadata.selection.names == [
        "model.blocks.0.ffn.key",
        "model.blocks.0.ffn.value",
        "model.blocks.1.ffn.key",
        "model.blocks.1.ffn.value",
    ]
    assert [
        name for name, _ in match_named_modules(model, {"Linear"}, ignore)
    ] == metadata.selection.names
    assert metadata.protections[0].names == ["model.blocks.0.att.value"]
    assert "produces v_first" in metadata.protections[0].reason
    assert metadata.protections[2].names == [
        "model.blocks.1.att.v1",
        "model.blocks.1.att.v2",
    ]
    assert metadata.protections[2].kind == "module"
    assert metadata.protections[3].names == ["model.blocks.1.att.v0"]
    assert metadata.protections[3].kind == "tensor"
    recurrent = metadata.protections[-1]
    assert recurrent.kind == "tensor"
    assert "model.blocks.0.att.x_r" in recurrent.names
    assert "model.blocks.0.ffn.x_k" in recurrent.names
    assert "model.blocks.1.att.r_k" in recurrent.names
    assert "model.blocks.1.ffn.x_k" in recurrent.names
    protected_modules = {
        name
        for decision in metadata.protections
        if decision.kind == "module"
        for name in decision.names
    }
    assert "model.embeddings" in protected_modules
    assert "model.ln_out" in protected_modules
    assert "model.blocks.0.ln0" in protected_modules
    assert "model.blocks.1.att.ln_x" in protected_modules
    recorded_names = metadata.selection.names + [
        name for decision in metadata.protections for name in decision.names
    ]
    assert all(name == "head" or name.startswith("model.") for name in recorded_names)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation, error_match",
    [
        (
            lambda model: setattr(model, "config", SimpleNamespace(model_type="llama")),
            "Rwkv7Config",
        ),
        (
            lambda model: setattr(model, "base_model_prefix", "rwkv7"),
            "base_model_prefix",
        ),
        (
            lambda model: delattr(model, "model"),
            "base_model.*registered",
        ),
        (
            lambda model: delattr(model.model.blocks[1].att, "v2"),
            r"blocks\.1\.att\.v2",
        ),
        (
            lambda model: setattr(
                model.model.blocks[0], "unexpected", torch.nn.Linear(8, 8)
            ),
            "non-standard Linear",
        ),
    ],
)
def test_rwkv7_policy_decision_table_fails_closed(mutation, error_match):
    model = _Rwkv7ForCausalLM()
    mutation(model)

    with pytest.raises(ValueError, match=error_match):
        _apply_policy(model)


@pytest.mark.unit
def test_rwkv7_policy_accepts_fused_embedding_norm_identity():
    model = _Rwkv7ForCausalLM()
    model.config.embedding_layer_norm_fused = True
    model.model.blocks[0].ln0 = torch.nn.Identity()

    _, metadata = _apply_policy(model)

    protected_modules = {
        name
        for decision in metadata.protections
        if decision.kind == "module"
        for name in decision.names
    }
    assert "model.embeddings" in protected_modules
    assert "model.blocks.0.ln0" not in protected_modules


@pytest.mark.unit
def test_rwkv7_policy_rejects_before_quantization_is_applied(monkeypatch):
    model = _Rwkv7ForCausalLM()
    del model.model.blocks[1].att.v2
    state = State()
    state.update(model=model, device="cpu")
    modifier = QuantizationModifier(scheme="W8A8", target_policy="rwkv7")
    applied = False

    def _record_apply(*args, **kwargs):
        nonlocal applied
        applied = True

    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.quantization.mixin."
        "apply_quantization_config",
        _record_apply,
    )

    with pytest.raises(ValueError, match=r"blocks\.1\.att\.v2"):
        modifier.on_initialize(state)

    assert applied is False


@pytest.mark.unit
def test_rwkv7_policy_metadata_recipe_round_trip():
    model = _Rwkv7ForCausalLM()
    modifier = QuantizationModifier(scheme="W8A8", target_policy="rwkv7")
    modifier._apply_target_policy(model)

    recipe_yaml = Recipe.from_modifiers(modifier).yaml()
    restored = Recipe.create_instance(recipe_yaml).modifiers[0]

    assert restored.target_policy == "rwkv7"
    assert restored.target_policy_metadata == modifier.target_policy_metadata
    assert "base_model_prefix: model" in recipe_yaml
    assert "produces v_first" in recipe_yaml
    assert "model.blocks.1.att.v2" in recipe_yaml
    assert "rwkv7.blocks" not in recipe_yaml


def _tiny_standard_rwkv7():
    from transformers import Rwkv7Config
    from transformers.models.rwkv7 import Rwkv7ForCausalLM

    return Rwkv7ForCausalLM(
        Rwkv7Config(
            vocab_size=32,
            context_length=16,
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            head_size=8,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "candidate",
        "algorithm",
        "weight_dtype",
        "weight_group_size",
        "input_dtype",
        "input_scale",
        "protection_profile",
        "candidate_role",
        "runtime_requirement",
    ),
    [
        (
            "nvfp4-w4a4",
            "NVFP4",
            "float4",
            16,
            "float4",
            "dynamic_local",
            "critical-high",
            "nvfp4-primary",
            "blackwell-sm120",
        ),
        (
            "nvfp4-w4a16",
            "NVFP4",
            "float4",
            16,
            "float16",
            "none",
            "critical-high",
            "nvfp4-weight-only-baseline",
            "blackwell-sm120",
        ),
        (
            "nvfp4-w4a16-protection-ablation",
            "NVFP4",
            "float4",
            16,
            "float16",
            "none",
            "v-first-dataflow",
            "nvfp4-protection-ablation",
            "blackwell-sm120",
        ),
        (
            "w8a16-low-rank-critical-high",
            "INT8",
            "int8",
            32,
            "float16",
            "none",
            "low-rank-w8-critical-high",
            "low-rank-w8-diagnostic",
            "portable-int8",
        ),
    ],
)
def test_tiny_standard_rwkv7_builds_closed_candidate_recipe(
    candidate,
    algorithm,
    weight_dtype,
    weight_group_size,
    input_dtype,
    input_scale,
    protection_profile,
    candidate_role,
    runtime_requirement,
    real_rwkv7_types,
):
    model = _tiny_standard_rwkv7()

    modifier = build_rwkv7_quantization_recipe(model, candidate)
    loader = modifier.target_policy_metadata.recipe

    assert loader.candidate_order == [
        "nvfp4-w4a4",
        "nvfp4-w4a16",
        "nvfp4-w4a16-protection-ablation",
        "w8a16-low-rank-critical-high",
    ]
    assert loader.algorithm == algorithm
    assert loader.weight_dtype == weight_dtype
    assert loader.weight_group_size == weight_group_size
    assert loader.input_dtype == input_dtype
    assert loader.input_scale == input_scale
    assert loader.protection_profile == protection_profile
    assert loader.candidate_role == candidate_role
    assert loader.runtime_requirement == runtime_requirement
    assert modifier.target_policy_metadata.protection_profile == protection_profile
    assert loader.targets == modifier.target_policy_metadata.selection.names
    assert loader.quantization_applied is False
    assert not any(hasattr(module, "quantization_scheme") for module in model.modules())


@pytest.mark.unit
@pytest.mark.parametrize(
    "candidate",
    [
        "nvfp4-w4a4",
        "nvfp4-w4a16",
        "nvfp4-w4a16-protection-ablation",
    ],
)
def test_nvfp4_recipe_survives_second_policy_resolution(
    candidate,
    real_rwkv7_types,
):
    model = _tiny_standard_rwkv7()
    modifier = build_rwkv7_quantization_recipe(model, candidate)
    recipe = modifier.target_policy_metadata.recipe
    state = State()
    state.update(model=model, device="cpu")

    modifier.on_initialize(state)

    assert modifier.target_policy_metadata.recipe == recipe
    assert modifier.target_policy_metadata.recipe.targets == (
        modifier.target_policy_metadata.selection.names
    )


@pytest.mark.unit
def test_protection_ablation_exactly_matches_vllm_nvfp4_consumer_matrix(
    real_rwkv7_types,
):
    model = _tiny_standard_rwkv7()
    modifier = build_rwkv7_quantization_recipe(model, "nvfp4-w4a16-protection-ablation")
    metadata = modifier.target_policy_metadata

    assert modifier.ignore == [
        r"re:^model\.blocks\.0\.att\.value$",
        r"re:^model\.blocks\.\d+\.ffn\.(?:key|value)$",
        "head",
    ]
    assert metadata.selection.names == [
        *(
            f"model.blocks.0.att.{name}"
            for name in ("w1", "w2", "a1", "a2", "g1", "g2")
        ),
        *(f"model.blocks.0.att.{name}" for name in ("receptance", "key", "output")),
        *(
            f"model.blocks.1.att.{name}"
            for name in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")
        ),
        *(
            f"model.blocks.1.att.{name}"
            for name in ("receptance", "key", "value", "output")
        ),
    ]
    assert len(metadata.selection.names) == 9 + 12 * (2 - 1)
    assert not any(".ffn." in name for name in metadata.selection.names)
    assert [
        name for name, _ in match_named_modules(model, {"Linear"}, modifier.ignore)
    ] == metadata.selection.names
    protected_modules = {
        name
        for decision in metadata.protections
        if decision.kind == "module"
        for name in decision.names
    }
    assert {
        "model.blocks.0.att.value",
        "model.blocks.0.ffn.key",
        "model.blocks.1.ffn.value",
        "head",
    } <= protected_modules
    assert set(metadata.selection.names).isdisjoint(protected_modules)
    protected_tensors = {
        name
        for decision in metadata.protections
        if decision.kind == "tensor"
        for name in decision.names
    }
    assert "model.blocks.1.att.v0" in protected_tensors
    assert "model.blocks.1.att.v2" not in protected_tensors


@pytest.mark.unit
def test_low_rank_w8_selects_standard_wag_linears_but_protects_v_first(
    real_rwkv7_types,
):
    model = _tiny_standard_rwkv7()
    modifier = build_rwkv7_quantization_recipe(
        model,
        "w8a16-low-rank-critical-high",
    )
    metadata = modifier.target_policy_metadata

    low_rank_modules = [
        f"model.blocks.{layer}.att.{name}"
        for layer in range(2)
        for name in ("w1", "w2", "a1", "a2", "g1", "g2")
    ]
    assert set(low_rank_modules) <= set(metadata.selection.names)
    assert all(
        isinstance(model.get_submodule(name), torch.nn.Linear)
        and model.get_submodule(name).bias is None
        for name in low_rank_modules
    )
    assert all(f"{name}.weight" in model.state_dict() for name in low_rank_modules)
    assert not any(name.endswith((".v1", ".v2")) for name in metadata.selection.names)
    protected_modules = {
        name
        for decision in metadata.protections
        if decision.kind == "module"
        for name in decision.names
    }
    protected_tensors = {
        name
        for decision in metadata.protections
        if decision.kind == "tensor"
        for name in decision.names
    }
    assert "model.blocks.0.att.value" in protected_modules
    assert {
        "model.blocks.1.att.v1",
        "model.blocks.1.att.v2",
    } <= protected_modules
    assert "model.blocks.1.att.v0" in protected_tensors
    assert metadata.recipe.low_rank_weight_dtype == "int8"
    assert modifier.bypass_divisibility_checks is False


@pytest.mark.unit
def test_tiny_standard_rwkv7_recipe_and_metadata_save_reload(
    tmp_path, real_rwkv7_types
):
    modifier = build_rwkv7_quantization_recipe(_tiny_standard_rwkv7())
    recipe_yaml = Recipe.from_modifiers(modifier).yaml()
    restored = Recipe.create_instance(recipe_yaml).modifiers[0]
    metadata_path = tmp_path / "rwkv7_quantization_metadata.json"
    metadata_path.write_text(
        modifier.target_policy_metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )
    loaded = QuantizationTargetPolicyMetadata.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    )

    assert restored.target_policy_metadata == modifier.target_policy_metadata
    assert loaded == modifier.target_policy_metadata
    assert loaded.selection.names == loaded.recipe.targets
    assert {name for protection in loaded.protections for name in protection.names}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("candidate", "versions", "message"),
    [
        ("nvfp4-unknown", None, "unsupported RWKV-7 quantization candidate"),
        (
            "nvfp4-w4a4",
            {"compressed_tensors": "unknown", "transformers": "5.15.0.dev0"},
            "validated framework versions",
        ),
    ],
)
def test_rwkv7_recipe_rejects_unknown_candidate_or_framework(
    candidate, versions, message, real_rwkv7_types
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        build_rwkv7_quantization_recipe(
            _tiny_standard_rwkv7(),
            candidate,
            framework_versions=versions,
        )


@pytest.mark.unit
def test_tiny_standard_rwkv7_recipe_rejects_target_drift_before_quantization(
    real_rwkv7_types,
):
    model = _tiny_standard_rwkv7()
    model.model.blocks[0].unexpected = torch.nn.Linear(16, 16)

    with pytest.raises(ValueError, match="non-standard Linear"):
        build_rwkv7_quantization_recipe(model)
