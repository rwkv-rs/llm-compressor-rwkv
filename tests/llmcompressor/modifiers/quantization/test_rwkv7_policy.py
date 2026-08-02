"""CPU tests for the fail-closed RWKV-7 quantization target policy."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.utils import match_named_modules

from llmcompressor.core import State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.rwkv7 import (
    apply_rwkv7_target_policy,
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


@pytest.fixture(autouse=True)
def _standard_rwkv7_types(monkeypatch):
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
    recorded_names = metadata.selection.names + [
        name for decision in metadata.protections for name in decision.names
    ]
    assert all(
        name == "head" or name.startswith("model.blocks.")
        for name in recorded_names
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation, error_match",
    [
        (
            lambda model: setattr(
                model, "config", SimpleNamespace(model_type="llama")
            ),
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
