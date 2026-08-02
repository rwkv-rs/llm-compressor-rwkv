"""Contracts for formal RWKV-7 checkpoint candidate execution."""

import hashlib
import json

import pytest

from llmcompressor.modifiers.quantization.rwkv7 import (
    RWKV7CheckpointContract,
    _load_calibration_records,
    build_rwkv7_artifact_contract,
    build_rwkv7_quantization_recipe,
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
