import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from llmcompressor.modifiers.quantization.rwkv7 import (
    RWKV7CheckpointContract,
    RWKV7TransformersProvenance,
    _fresh_reload_generate_script,
    _validate_rwkv7_transformers_source_provenance,
    quantize_rwkv7_oneshot,
)


@pytest.fixture
def owned_process_tmpdir(tmp_path, monkeypatch):
    process_tmpdir = tmp_path / "process-tmp"
    process_tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(process_tmpdir))
    monkeypatch.setattr(tempfile, "tempdir", str(process_tmpdir))
    yield process_tmpdir
    shutil.rmtree(process_tmpdir, ignore_errors=True)


def _model(device="cuda"):
    from transformers import Rwkv7Config
    from transformers.models.rwkv7 import Rwkv7ForCausalLM

    return (
        Rwkv7ForCausalLM(
            Rwkv7Config(
                vocab_size=32,
                context_length=16,
                hidden_size=16,
                num_hidden_layers=2,
                intermediate_size=32,
                head_size=8,
            )
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )


def _synthetic_runtime_provenance(*_args):
    return RWKV7TransformersProvenance(
        repository="https://github.com/rwkv-rs/transformers-rwkv.git",
        revision="2696927df9363b5fa175076bb827ba4da2c4e581",
        installation_source="editable-git",
        editable=True,
        operator_runtime={"scope": "synthetic-unit-boundary"},
    )


@pytest.mark.unit
def test_fresh_reload_program_is_valid_python():
    script = _fresh_reload_generate_script()
    compile(script, "<rwkv7-fresh-reload>", "exec")
    assert "_validate_rwkv7_transformers_source_provenance(" in script
    assert "'transformers_provenance': None" not in script


@pytest.mark.unit
def test_fresh_reload_program_fails_closed_under_python_optimize(tmp_path):
    script = _fresh_reload_generate_script()
    parsed = ast.parse(script)
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(parsed))
    assert script.count("trust_remote_code=False") == 2
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "rwkv7",
                "architectures": ["Rwkv7ForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_rwkv7.Rwkv7Config",
                },
            }
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONOPTIMIZE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            "20260801",
            "[1, 2, 3, 4]",
            "1",
            "1",
            "1",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "native Transformers classes without auto_map" in result.stderr


@pytest.mark.unit
def test_execution_falls_back_only_in_closed_order(tmp_path, monkeypatch):
    calls = []

    def oneshot(*, model, recipe, **kwargs):
        candidate = recipe.target_policy_metadata.recipe.candidate
        calls.append(candidate)
        if candidate == "nvfp4-w4a4":
            raise RuntimeError("W4A4 unavailable")
        return model

    monkeypatch.setattr("llmcompressor.oneshot", oneshot)
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7.audit_rwkv7_quantized_checkpoint",
        lambda *args, **kwargs: {"format": "nvfp4-pack-quantized"},
    )
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7."
        "validate_rwkv7_transformers_provenance",
        _synthetic_runtime_provenance,
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="{}\n",
            stderr="",
        ),
    )
    _, metadata = quantize_rwkv7_oneshot(
        lambda: _model("cpu"),
        tmp_path,
        calibration_dataset=object(),
        processor=object(),
    )

    assert calls == ["nvfp4-w4a4", "nvfp4-w4a16"]
    assert metadata["candidate"] == "nvfp4-w4a16"
    assert metadata["quantization_applied"] is True
    assert metadata["quantization_runtime"]["scope"] == "llmcompressor-oneshot"
    assert metadata["failures"][0]["error"] == "W4A4 unavailable"


@pytest.mark.unit
def test_formal_execution_rejects_a_failed_fresh_process_reload(tmp_path, monkeypatch):
    monkeypatch.setattr("llmcompressor.oneshot", lambda *, model, **kwargs: model)
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7.audit_rwkv7_quantized_checkpoint",
        lambda *args, **kwargs: {"format": "nvfp4-pack-quantized"},
    )
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7."
        "validate_rwkv7_transformers_provenance",
        _synthetic_runtime_provenance,
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="fresh process could not load",
        ),
    )

    with pytest.raises(RuntimeError, match="fresh_reload_generate"):
        quantize_rwkv7_oneshot(
            lambda: _model("cpu"),
            tmp_path,
            calibration_dataset=None,
            processor=None,
            forced_candidate="nvfp4-w4a16",
            checkpoint_contract=RWKV7CheckpointContract(),
        )

    evidence = json.loads(
        (tmp_path / "nvfp4-w4a16" / "rwkv7_quantization_execution.json").read_text()
    )
    assert evidence["fresh_reload"]["passed"] is False
    assert evidence["failures"][-1]["stage"] == "fresh_reload_generate"


@pytest.mark.unit
def test_source_only_execution_rejects_a_failed_fresh_process_load(
    tmp_path,
    monkeypatch,
):
    source_provenance = _synthetic_runtime_provenance().model_copy(
        update={"operator_runtime": {}}
    )
    monkeypatch.setattr("llmcompressor.oneshot", lambda *, model, **kwargs: model)
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7.audit_rwkv7_quantized_checkpoint",
        lambda *args, **kwargs: {"format": "nvfp4-pack-quantized"},
    )
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7."
        "_validate_rwkv7_transformers_source_provenance",
        lambda *_args: source_provenance,
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="fresh process could not load",
        ),
    )

    with pytest.raises(RuntimeError, match="fresh_reload_load"):
        quantize_rwkv7_oneshot(
            lambda: _model("cpu"),
            tmp_path,
            calibration_dataset=None,
            processor=None,
            forced_candidate="nvfp4-w4a16",
            fresh_reload_mode="load-only",
            fresh_reload_device="cpu",
        )

    evidence = json.loads(
        (tmp_path / "nvfp4-w4a16" / "rwkv7_quantization_execution.json").read_text()
    )
    assert evidence["fresh_reload"]["passed"] is False
    assert evidence["failures"][-1]["stage"] == "fresh_reload_load"


@pytest.mark.integration
def test_nvfp4_w4a16_oneshot_save_and_fresh_direct_class_load(
    tmp_path,
    owned_process_tmpdir,
):
    try:
        observed_transformers = (
            _validate_rwkv7_transformers_source_provenance().model_dump(mode="json")
        )
    except RuntimeError as error:
        pytest.skip(f"requires the product-pinned Transformers revision: {error}")

    model, metadata = quantize_rwkv7_oneshot(
        lambda: _model("cpu"),
        tmp_path,
        calibration_dataset=None,
        processor=None,
        forced_candidate="nvfp4-w4a16",
        fresh_reload_mode="load-only",
        fresh_reload_device="cpu",
    )

    artifact_path = tmp_path / "nvfp4-w4a16"
    assert artifact_path.joinpath("model.safetensors").is_file()
    assert metadata["candidate"] == "nvfp4-w4a16"
    assert metadata["quantization_applied"] is True
    assert metadata["provenance_scope"] == "serialization-only"
    assert metadata["artifact_contract"]["runtime_provenance"] == (
        observed_transformers
    )
    assert metadata["audit"]["format"] == "nvfp4-pack-quantized"
    assert metadata["audit"]["input_quantized"] is False
    assert metadata["audit"]["legacy_weight_aliases"] == []
    assert metadata["audit"]["protected_parameter_values_verified"] is True
    assert metadata["audit"]["operator_runtime_provenance_required"] is False
    protection_audit = metadata["protection_audit"]
    assert protection_audit["passed"] is True
    assert protection_audit["module_identity_preserved"] is True
    assert protection_audit["parameter_identity_preserved"] is True
    assert protection_audit["parameter_ownership_preserved"] is True
    assert protection_audit["parameter_values_preserved"] is True
    assert protection_audit["parameter_count"] == len(
        protection_audit["parameter_sha256"]
    )
    assert metadata["fresh_reload"]["passed"] is True
    assert metadata["fresh_reload"]["mode"] == "load-only"
    assert metadata["fresh_reload"]["device"] == "cpu"
    fresh_evidence = metadata["fresh_reload"]["evidence"]
    assert fresh_evidence["execution_mode"] == "load-only"
    assert fresh_evidence["standard_generate"] == {
        "passed": False,
        "executed": False,
    }
    assert fresh_evidence["transformers_provenance"] == observed_transformers
    strict_load = fresh_evidence["standard_linear_load"]
    assert strict_load["loader"] == "Rwkv7ForCausalLM.from_pretrained"
    assert strict_load["strict_loading_info"] is True
    assert strict_load["runtime_float_weights_restored"] is True
    assert strict_load["protected_parameter_values_verified"] is True
    assert strict_load["vllm_metadata_validated"] is True
    assert strict_load["quantized_scheme_count"] == 4
    assert all(
        getattr(model.get_submodule(name), "quantization_scheme", None) is not None
        for name in metadata["artifact_contract"]["vllm"]["quantized_modules"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fresh_reload_mode": "metadata-only"}, "fresh reload mode"),
        ({"fresh_reload_device": "cuda:any"}, "fresh reload device"),
        (
            {
                "checkpoint_contract": RWKV7CheckpointContract(),
                "fresh_reload_mode": "load-only",
            },
            "formal RWKV-7 checkpoint execution requires fresh forward/generate",
        ),
    ],
)
def test_execution_rejects_invalid_fresh_process_boundary(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        quantize_rwkv7_oneshot(
            lambda: pytest.fail("validation must run before model construction"),
            tmp_path,
            calibration_dataset=None,
            processor=None,
            forced_candidate="nvfp4-w4a16",
            **kwargs,
        )


@pytest.mark.skipif(
    os.environ.get("LLMCOMPRESSOR_RWKV7_GPU_TEST") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 12,
    reason="requires opt-in Blackwell NVFP4 execution",
)
@pytest.mark.parametrize("candidate", ["nvfp4-w4a4", "nvfp4-w4a16"])
@pytest.mark.integration
def test_gb10_real_nvfp4_checkpoint_has_packed_tensors_and_forward(
    tmp_path, owned_process_tmpdir, candidate
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>"))
    processor = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<unk>"
    )
    dataset = DataLoader(
        [
            {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
                "attention_mask": torch.ones((1, 8), dtype=torch.long),
            }
        ],
        batch_size=None,
    )
    model, metadata = quantize_rwkv7_oneshot(
        _model,
        tmp_path,
        calibration_dataset=dataset,
        processor=processor,
        forced_candidate=candidate,
    )

    assert metadata["candidate"] == candidate
    assert metadata["forced_candidate"] == candidate
    assert metadata["quantization_applied"] is True
    assert metadata["audit"]["format"] == "nvfp4-pack-quantized"
    assert metadata["audit"]["artifact_contract_serialized"] is True
    assert len(metadata["audit"]["targets"]) == 4
    assert metadata["audit"]["input_quantized"] is (candidate == "nvfp4-w4a4")
    assert metadata["cell_forward"] == {
        "passed": True,
        "output_shape": [1, 4, 16],
        "state_shape": [1, 16],
    }
    saved = json.loads(
        (tmp_path / candidate / "rwkv7_quantization_execution.json").read_text()
    )
    assert saved == metadata
    assert metadata["fresh_reload"]["passed"] is True
    assert metadata["fresh_reload"]["returncode"] == 0
    assert metadata["fresh_reload"]["source_owner"] == "Transformers RWKV7 loader"
    assert (
        "standard compressed-tensors dequantization path"
        in metadata["fresh_reload"]["regression_expectation"]
    )
    reload_evidence = metadata["fresh_reload"]["evidence"]
    assert reload_evidence["dtype"] == "torch.bfloat16"
    assert reload_evidence["logits_dtype"] == "torch.bfloat16"
    assert reload_evidence["quantized_module_count"] == 4
    assert reload_evidence["protected_module_count"] == 32
    assert reload_evidence["protected_state_tensor_count"] == 25
    assert reload_evidence["protected_parameter_count"] == 65
    assert reload_evidence["artifact_contract_validated"] is True
    runtime = reload_evidence["runtime_measurement"]
    assert runtime["scope"] == "fresh-process-transformers-generate-diagnostic"
    assert runtime["canonical_performance_acceptance"] is False
    assert runtime["generate"]["warmup_runs"] == 1
    assert runtime["generate"]["timed_runs"] == 3
    artifact_contract = metadata["artifact_contract"]
    assert artifact_contract["formal_checkpoint"] is False
    assert artifact_contract["formal_evaluation"] is False
    assert artifact_contract["vllm"]["source_format"] == "standard_hf"
    assert artifact_contract["vllm"]["legacy_pth_direct_load"] is False
    assert artifact_contract["vllm"]["linear_weight_suffix"] == "weight"
    assert artifact_contract["vllm"]["linear_weight_layout"] == "out-in"
    expected_capability = (
        "vllm-rwkv-nvfp4-w4a4" if candidate == "nvfp4-w4a4" else "vllm-rwkv-nvfp4-w4a16"
    )
    assert artifact_contract["vllm"]["vllm_consumer_requirement"] == (
        expected_capability
    )
    expected_capabilities = ["transformers-rwkv-compressed-tensors"]
    if candidate == "nvfp4-w4a16":
        expected_capabilities.append(expected_capability)
    assert artifact_contract["vllm"]["consumer_capabilities"] == (expected_capabilities)
    assert artifact_contract["vllm"]["vllm_consumer_revision"] == (
        None
        if candidate == "nvfp4-w4a4"
        else "88b992bbc73e8b904ae672dfd39396b6dd0d6ea4"
    )
    assert artifact_contract["vllm"]["quantized_low_rank_modules"] == []
    assert artifact_contract["vllm"]["protected_v_first_linear_modules"] == [
        "model.blocks.1.att.v1",
        "model.blocks.1.att.v2",
    ]
    assert "model.blocks.0.att.value" in artifact_contract["vllm"]["protected_modules"]
    generate_evidence = reload_evidence["standard_generate"]
    assert {
        key: value for key, value in generate_evidence.items() if key != "generated_ids"
    } == {
        "passed": True,
        "use_cache": True,
        "seed": 20260801,
        "prompt_ids": [1, 2, 3, 4],
        "max_new_tokens": 4,
    }
    assert len(generate_evidence["generated_ids"]) == 1
    assert generate_evidence["generated_ids"][0][:4] == [1, 2, 3, 4]
    assert len(generate_evidence["generated_ids"][0]) == 8
