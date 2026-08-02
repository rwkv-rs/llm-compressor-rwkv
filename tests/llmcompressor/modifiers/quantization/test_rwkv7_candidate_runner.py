"""Contracts for formal RWKV-7 checkpoint candidate execution."""

import hashlib
import json
import subprocess
import sys

import pytest
import torch

from llmcompressor.modifiers.quantization.rwkv7 import (
    RWKV7CheckpointContract,
    _apply_rwkv7_low_rank_w8,
    _audit_rwkv7_low_rank_w8,
    _fresh_reload_generate_script,
    _iter_rwkv7_low_rank_w8_dequantized,
    _load_calibration_records,
    _serialize_rwkv7_low_rank_w8,
    build_rwkv7_artifact_contract,
    build_rwkv7_quantization_recipe,
    validate_rwkv7_low_rank_standard_load,
    verify_rwkv7_checkpoint,
)


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
def test_artifact_contract_pins_fork_standard_names_and_v_first_protection():
    modifier = build_rwkv7_quantization_recipe(_tiny_standard_rwkv7())

    contract = build_rwkv7_artifact_contract(
        modifier.target_policy_metadata,
        "nvfp4-w4a4",
        checkpoint=RWKV7CheckpointContract(),
    )

    assert contract.repository.upstream_oid == (
        "28c9c76b74cdd47076f95d012227482d22a8f365"
    )
    assert contract.repository.fork_repository == (
        "https://github.com/rwkv-rs/llm-compressor-rwkv.git"
    )
    assert contract.checkpoint.sha256 == (
        "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c"
    )
    assert contract.vllm.architecture == "Rwkv7ForCausalLM"
    assert contract.vllm.embedding_name == "model.embeddings.weight"
    assert contract.vllm.legacy_pth_direct_load is False
    assert "model.blocks.0.att.value" in contract.vllm.protected_modules
    assert "model.blocks.1.att.v0" in contract.vllm.protected_tensors
    assert contract.vllm.quantized_modules == [
        "model.blocks.0.ffn.key",
        "model.blocks.0.ffn.value",
        "model.blocks.1.ffn.key",
        "model.blocks.1.ffn.value",
    ]
    assert contract.formal_checkpoint is True
    assert contract.formal_evaluation is False


@pytest.mark.unit
def test_low_rank_w8_intermediate_sidecar_and_standard_load_fail_closed(
    tmp_path,
):
    from transformers import AutoModelForCausalLM

    model = _tiny_standard_rwkv7().eval()
    modifier = build_rwkv7_quantization_recipe(
        model,
        "w8a16-low-rank-critical-high",
    )
    contract = build_rwkv7_artifact_contract(
        modifier.target_policy_metadata,
        "w8a16-low-rank-critical-high",
    )
    low_rank = contract.vllm.low_rank_w8
    assert low_rank is not None
    assert contract.vllm.quantization_format == "pack-quantized"
    assert contract.vllm.quantized_modules == [
        "model.blocks.0.ffn.key",
        "model.blocks.0.ffn.value",
        "model.blocks.1.ffn.key",
        "model.blocks.1.ffn.value",
    ]
    assert not set(contract.vllm.quantized_modules) & set(low_rank.parameter_names)
    assert low_rank.artifact_role == "intermediate-serialization-only"
    assert low_rank.standard_transformers_load_supported is False
    assert low_rank.standard_vllm_load_supported is False
    assert low_rank.required_parameter_ownership == (
        "standard-quantizable-module-weight"
    )
    fresh_reload_script = _fresh_reload_generate_script()
    assert "restore_rwkv7_low_rank_w8" not in fresh_reload_script
    assert "output_loading_info=True" in fresh_reload_script
    assert "validate_rwkv7_low_rank_standard_load" in fresh_reload_script
    assert "model.blocks.1.att.v1" in low_rank.protected_v_first_parameter_names
    with torch.no_grad():
        for index, name in enumerate(low_rank.parameter_names, start=1):
            model.get_parameter(name).uniform_(-0.25 * index, 0.25 * index)
    protected_before = {
        name: model.get_parameter(name).detach().clone()
        for name in low_rank.protected_v_first_parameter_names
    }

    runtime = _apply_rwkv7_low_rank_w8(model, low_rank)
    expected = {
        name: model.get_parameter(name).detach().clone()
        for name in low_rank.parameter_names
    }
    with torch.inference_mode():
        logits = model(torch.tensor([[1, 2, 3]]), use_cache=True).logits
    assert logits.shape == (1, 3, 32)
    assert torch.isfinite(logits).all()
    for name, tensor in protected_before.items():
        assert torch.equal(model.get_parameter(name), tensor)

    setattr(
        model.config,
        "rwkv7_quantization_metadata",
        contract.model_dump(mode="json"),
    )
    model.save_pretrained(tmp_path, safe_serialization=True)
    serialization = _serialize_rwkv7_low_rank_w8(tmp_path, low_rank)
    audit = _audit_rwkv7_low_rank_w8(tmp_path, low_rank)

    assert runtime["parameter_count"] == 12
    assert runtime["max_abs_error"] > 0
    assert serialization["parameter_count"] == 12
    assert serialization["packed_bytes"] + serialization["scale_bytes"] < sum(
        tensor.numel() * tensor.element_size() for tensor in expected.values()
    )
    assert audit["parameter_names"] == low_rank.parameter_names
    assert audit["all_finite"] is True
    assert audit["v_first_protected"] is True
    assert audit["artifact_role"] == "intermediate-serialization-only"
    assert audit["standard_transformers_load_supported"] is False
    assert audit["standard_vllm_load_supported"] is False
    intermediate = dict(
        _iter_rwkv7_low_rank_w8_dequantized(
            tmp_path,
            low_rank,
            dtype=torch.float32,
        )
    )
    for name, tensor in expected.items():
        assert torch.equal(intermediate[name], tensor.float())

    _, loading_info = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        output_loading_info=True,
    )
    with pytest.raises(
        RuntimeError,
        match="standard Transformers loader did not restore RWKV-7 low-rank W8",
    ):
        validate_rwkv7_low_rank_standard_load(loading_info, low_rank)
    assert set(low_rank.parameter_names) <= set(loading_info["missing_keys"])

    fresh_process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from transformers import AutoConfig, AutoModelForCausalLM
from llmcompressor.modifiers.quantization.rwkv7 import (
    validate_rwkv7_low_rank_standard_load,
)

config = AutoConfig.from_pretrained(sys.argv[1])
contract = config.rwkv7_quantization_metadata['vllm']['low_rank_w8']
_, loading_info = AutoModelForCausalLM.from_pretrained(
    sys.argv[1],
    output_loading_info=True,
)
validate_rwkv7_low_rank_standard_load(loading_info, contract)
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert fresh_process.returncode != 0
    assert (
        "standard Transformers loader did not restore RWKV-7 low-rank W8"
        in fresh_process.stderr
    )


@pytest.mark.unit
def test_calibration_loader_binds_digest_and_rejects_invalid_token_ids(tmp_path):
    calibration_path = tmp_path / "calibration.jsonl"
    calibration_path.write_text(
        "\n".join(
            [
                json.dumps({"input_ids": [1, 2, 3]}),
                json.dumps({"input_ids": [4, 5, 6]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

    records, metadata = _load_calibration_records(
        calibration_path,
        expected_sha256=digest,
        max_samples=1,
        max_length=2,
        vocab_size=32,
    )

    assert records[0]["input_ids"].tolist() == [[1, 2]]
    assert records[0]["attention_mask"].tolist() == [[1, 1]]
    assert metadata["sha256"] == digest
    assert metadata["sample_count"] == 1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_calibration_records(
            calibration_path,
            expected_sha256="0" * 64,
            max_samples=1,
            max_length=2,
            vocab_size=32,
        )

    calibration_path.write_text(
        json.dumps({"input_ids": [32]}) + "\n", encoding="utf-8"
    )
    invalid_digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="invalid input_ids"):
        _load_calibration_records(
            calibration_path,
            expected_sha256=invalid_digest,
            max_samples=1,
            max_length=2,
            vocab_size=32,
        )


@pytest.mark.unit
def test_formal_checkpoint_verifier_rejects_an_unpinned_filename(tmp_path):
    checkpoint_path = tmp_path / "other.pth"
    checkpoint_path.write_bytes(b"not the pinned checkpoint")

    with pytest.raises(ValueError, match="only accepts the pinned checkpoint"):
        verify_rwkv7_checkpoint(checkpoint_path)
