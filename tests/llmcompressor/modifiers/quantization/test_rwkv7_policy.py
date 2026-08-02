"""CPU tests for the fail-closed RWKV-7 quantization target policy."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.utils import match_named_modules

from llmcompressor.core import State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.rwkv7 import (
    QuantizationTargetPolicyMetadata,
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
        ):
            setattr(self, name, torch.nn.Parameter(torch.zeros(hidden_size)))
        if layer_id > 0:
            for name in ("v0", "v1", "v2"):
                setattr(self, name, torch.nn.Parameter(torch.zeros(hidden_size)))
        self.receptance = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class _ChannelMix(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.x_k = torch.nn.Parameter(torch.zeros(hidden_size))
        self.key = torch.nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.value = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)


class _Block(torch.nn.Module):
    def __init__(self, layer_id: int, hidden_size: int):
        super().__init__()
        self.att = _TimeMix(layer_id, hidden_size)
        self.ffn = _ChannelMix(hidden_size)


class _Rwkv7Model(torch.nn.Module):
    def __init__(self, config: _Rwkv7Config, hidden_size: int):
        super().__init__()
        self.config = config
        self.blocks = torch.nn.ModuleList(
            [
                _Block(layer_id, hidden_size)
                for layer_id in range(config.num_hidden_layers)
            ]
        )


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
        r"re:^model\.blocks\.\d+\.att\.(receptance|key|value|output)$",
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
        "model.blocks.1.att.v0",
        "model.blocks.1.att.v1",
        "model.blocks.1.att.v2",
    ]
    recurrent = metadata.protections[-1]
    assert recurrent.kind == "tensor"
    assert "model.blocks.0.att.x_r" in recurrent.names
    assert "model.blocks.1.att.r_k" in recurrent.names
    recorded_names = metadata.selection.names + [
        name for decision in metadata.protections for name in decision.names
    ]
    assert all(
        name == "head" or name.startswith("model.blocks.") for name in recorded_names
    )


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
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            head_size=8,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("candidate", "input_dtype", "input_scale"),
    [
        ("nvfp4-w4a4", "float4", "dynamic_local"),
        ("nvfp4-w4a16", "float16", "none"),
    ],
)
def test_tiny_standard_rwkv7_builds_closed_nvfp4_recipe(
    candidate, input_dtype, input_scale, real_rwkv7_types
):
    model = _tiny_standard_rwkv7()

    modifier = build_rwkv7_quantization_recipe(model, candidate)
    loader = modifier.target_policy_metadata.recipe

    assert loader.candidate_order == ["nvfp4-w4a4", "nvfp4-w4a16"]
    assert loader.algorithm == "NVFP4"
    assert loader.weight_dtype == "float4"
    assert loader.weight_group_size == 16
    assert loader.weight_scale_dtype == "float8_e4m3fn"
    assert loader.input_dtype == input_dtype
    assert loader.input_scale == input_scale
    assert loader.targets == modifier.target_policy_metadata.selection.names
    assert loader.quantization_applied is False
    assert not any(hasattr(module, "quantization_scheme") for module in model.modules())


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
