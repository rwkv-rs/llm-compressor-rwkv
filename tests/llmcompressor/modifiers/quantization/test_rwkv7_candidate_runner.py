"""Contracts for formal RWKV-7 checkpoint candidate execution."""

import hashlib
import json
import subprocess
import sys

import pytest
import torch

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.quantization.rwkv7 import (
    RWKV7ArtifactContract,
    RWKV7CheckpointContract,
    RWKV7TransformersProvenance,
    _fresh_reload_generate_script,
    _load_calibration_records,
    audit_rwkv7_quantized_checkpoint,
    build_rwkv7_artifact_contract,
    build_rwkv7_quantization_recipe,
    validate_rwkv7_transformers_provenance,
    verify_rwkv7_checkpoint,
)


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


def _run_fresh_process_load(model_path, *, compressed: bool):
    provenance_import = ""
    provenance_check = "transformers_provenance_validated = False"
    quantization_import = ""
    quantization_argument = ""
    contract_checks = ""
    if compressed:
        provenance_import = """
from llmcompressor.modifiers.quantization.rwkv7 import (
    RWKV7RepositoryContract,
    validate_rwkv7_transformers_provenance,
)
"""
        provenance_check = """
with open(f'{sys.argv[1]}/config.json', encoding='utf-8') as config_handle:
    serialized_config = json.load(config_handle)
contract = serialized_config['rwkv7_quantization_metadata']
validate_rwkv7_transformers_provenance(
    RWKV7RepositoryContract.model_validate(contract['repository'])
)
transformers_provenance_validated = True
"""
        quantization_import = (
            "from transformers.utils.quantization_config import "
            "CompressedTensorsConfig"
        )
        quantization_argument = (
            "quantization_config=CompressedTensorsConfig(dequantize=True),"
        )
        contract_checks = """
missing = sorted(
    set(contract['vllm']['quantized_weight_names'])
    & set(loading_info['missing_keys'])
)
assert not missing, missing
quantized_low_rank = [
    model.get_submodule(name)
    for name in contract['vllm']['quantized_low_rank_modules']
]
protected_v_first = [
    model.get_submodule(name)
    for name in contract['vllm']['protected_v_first_linear_modules']
]
assert all(
    getattr(module, 'quantization_scheme', None) is not None
    for module in quantized_low_rank
)
assert all(torch.count_nonzero(module.weight) > 0 for module in quantized_low_rank)
assert all(
    getattr(module, 'quantization_scheme', None) is None
    for module in protected_v_first
)
quantized_low_rank_count = len(quantized_low_rank)
protected_v_first_count = len(protected_v_first)
"""
    else:
        contract_checks = """
missing = []
quantized_low_rank_count = 0
protected_v_first_count = 0
"""

    script = f"""
import json, sys, torch
{provenance_import}
{provenance_check}
from transformers import AutoConfig, AutoModelForCausalLM
{quantization_import}

config = AutoConfig.from_pretrained(sys.argv[1])
model, loading_info = AutoModelForCausalLM.from_pretrained(
    sys.argv[1],
    {quantization_argument}
    output_loading_info=True,
)
{contract_checks}
model.eval()
prompt = torch.tensor([[1, 2, 3]])
with torch.inference_mode():
    logits = model(prompt, use_cache=True).logits
    generated = model.generate(
        prompt,
        max_new_tokens=2,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=[],
    )
assert torch.isfinite(logits).all()
print(json.dumps({{
    'logits_shape': list(logits.shape),
    'generated_shape': list(generated.shape),
    'missing_quantized_weights': missing,
    'quantized_low_rank_module_count': quantized_low_rank_count,
    'protected_v_first_linear_module_count': protected_v_first_count,
    'transformers_provenance_validated': transformers_provenance_validated,
}}))
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(model_path)],
        capture_output=True,
        text=True,
    )


@pytest.mark.unit
def test_rwkv7_transformers_provenance_matches_exact_editable_fork():
    provenance = validate_rwkv7_transformers_provenance()

    assert provenance.repository == (
        "https://github.com/rwkv-rs/transformers-rwkv.git"
    )
    assert provenance.revision == (
        "2696927df9363b5fa175076bb827ba4da2c4e581"
    )
    assert provenance.installation_source == "editable-git"
    assert provenance.editable is True


@pytest.mark.unit
def test_rwkv7_transformers_provenance_rejects_registry_install(monkeypatch):
    class _RegistryDistribution:
        @staticmethod
        def read_text(filename):
            assert filename == "direct_url.json"
            return None

    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7."
        "importlib_metadata.distribution",
        lambda name: _RegistryDistribution(),
    )

    with pytest.raises(RuntimeError, match="registry-only installation"):
        validate_rwkv7_transformers_provenance()


@pytest.mark.unit
def test_rwkv7_transformers_provenance_rejects_unpinned_vcs_request(monkeypatch):
    import transformers

    class _BranchVcsDistribution:
        @staticmethod
        def read_text(filename):
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/rwkv-rs/transformers-rwkv.git",
                    "vcs_info": {
                        "vcs": "git",
                        "requested_revision": "main",
                        "commit_id": (
                            "2696927df9363b5fa175076bb827ba4da2c4e581"
                        ),
                    },
                }
            )

        @staticmethod
        def locate_file(filename):
            assert filename == "transformers/__init__.py"
            return transformers.__file__

    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7."
        "importlib_metadata.distribution",
        lambda name: _BranchVcsDistribution(),
    )

    with pytest.raises(RuntimeError, match="requested revision is not exact"):
        validate_rwkv7_transformers_provenance()


@pytest.mark.unit
def test_rwkv7_transformers_provenance_rejects_revision_drift(monkeypatch):
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7."
        "_installed_transformers_provenance",
        lambda: RWKV7TransformersProvenance(
            repository="https://github.com/rwkv-rs/transformers-rwkv.git",
            revision="0" * 40,
            installation_source="pep610-vcs",
            editable=False,
        ),
    )

    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate_rwkv7_transformers_provenance()


@pytest.mark.unit
def test_artifact_contract_pins_fork_standard_names_and_v_first_protection():
    modifier = build_rwkv7_quantization_recipe(_tiny_standard_rwkv7())
    assert modifier.target_policy_metadata.recipe.quantization_applied is False

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
    assert contract.repository.transformers_oid == (
        "2696927df9363b5fa175076bb827ba4da2c4e581"
    )
    assert contract.checkpoint.sha256 == (
        "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c"
    )
    assert contract.vllm.architecture == "Rwkv7ForCausalLM"
    assert contract.vllm.embedding_name == "model.embeddings.weight"
    assert contract.vllm.quantization_target_type == "Linear"
    assert contract.vllm.linear_weight_suffix == "weight"
    assert contract.vllm.linear_weight_layout == "out-in"
    assert contract.vllm.legacy_pth_direct_load is False
    assert contract.vllm.layer_zero_v_first_producer == (
        "model.blocks.0.att.value"
    )
    assert contract.vllm.protected_embedding_modules == ["model.embeddings"]
    assert "head" in contract.vllm.protected_linear_modules
    assert "model.blocks.0.ln0" in contract.vllm.protected_normalization_modules
    assert "model.blocks.1.ln0" not in (
        contract.vllm.protected_normalization_modules
    )
    assert "model.blocks.0.att.value" in contract.vllm.protected_modules
    assert "model.blocks.1.att.v0" in contract.vllm.protected_tensors
    assert "model.blocks.0.ffn.x_k" in contract.vllm.protected_state_tensors
    assert contract.vllm.quantized_modules == [
        "model.blocks.0.ffn.key",
        "model.blocks.0.ffn.value",
        "model.blocks.1.ffn.key",
        "model.blocks.1.ffn.value",
    ]
    assert contract.formal_checkpoint is True
    assert contract.formal_evaluation is False
    assert contract.target_policy.recipe.quantization_applied is True


@pytest.fixture
def standard_linear_w8_artifact(tmp_path):
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )

    model = _tiny_standard_rwkv7().eval()
    modifier = build_rwkv7_quantization_recipe(
        model,
        "w8a16-low-rank-critical-high",
    )
    contract = build_rwkv7_artifact_contract(
        modifier.target_policy_metadata,
        "w8a16-low-rank-critical-high",
    )
    assert contract.vllm.quantization_format == "pack-quantized"
    channel_mix_modules = [
        "model.blocks.0.ffn.key",
        "model.blocks.0.ffn.value",
        "model.blocks.1.ffn.key",
        "model.blocks.1.ffn.value",
    ]
    low_rank_modules = [
        f"model.blocks.{layer}.att.{name}"
        for layer in range(2)
        for name in ("w1", "w2", "a1", "a2", "g1", "g2")
    ]
    assert set(contract.vllm.quantized_modules) == set(
        [*channel_mix_modules, *low_rank_modules]
    )
    assert contract.vllm.quantized_low_rank_modules == low_rank_modules
    assert contract.vllm.low_rank_linear_modules == sorted(low_rank_modules)
    assert contract.vllm.protected_v_first_linear_modules == [
        "model.blocks.1.att.v1",
        "model.blocks.1.att.v2",
    ]
    assert contract.vllm.quantized_weight_names == [
        f"{name}.weight" for name in contract.vllm.quantized_modules
    ]
    fresh_reload_script = _fresh_reload_generate_script()
    assert "rwkv7_low_rank_w8" not in fresh_reload_script
    assert "output_loading_info=True" in fresh_reload_script
    assert "quantized_weight_names" in fresh_reload_script
    with torch.no_grad():
        for index, name in enumerate(low_rank_modules, start=1):
            model.get_submodule(name).weight.uniform_(
                -0.05 * index,
                0.05 * index,
            )
    protected_before = {
        name: model.get_submodule(name).weight.detach().clone()
        for name in contract.vllm.protected_v_first_linear_modules
    }

    state = State()
    state.update(model=model, device="cpu")
    modifier.on_initialize(state)
    modifier.on_calibration_start(
        state,
        Event(type_=EventType.CALIBRATION_START),
    )
    modifier.on_sequential_epoch_end(
        state,
        Event(type_=EventType.SEQUENTIAL_EPOCH_END),
        modules=list(model.modules()),
    )
    modifier.on_calibration_end(
        state,
        Event(type_=EventType.CALIBRATION_END),
    )
    result = model
    with torch.inference_mode():
        logits = result(torch.tensor([[1, 2, 3]]), use_cache=True).logits
    assert logits.shape == (1, 3, 32)
    assert torch.isfinite(logits).all()
    for name, tensor in protected_before.items():
        assert torch.equal(result.get_submodule(name).weight, tensor)

    setattr(
        result.config,
        "rwkv7_quantization_metadata",
        contract.model_dump(mode="json"),
    )
    modify_save_pretrained(result)
    result.save_pretrained(tmp_path, save_compressed=True)
    audit = audit_rwkv7_quantized_checkpoint(
        tmp_path,
        contract.vllm.quantized_modules,
        "w8a16-low-rank-critical-high",
        contract,
    )
    assert audit["standard_linear_ownership"] is True
    assert audit["targets"] == contract.vllm.quantized_modules
    assert audit["quantized_weight_names"] == contract.vllm.quantized_weight_names
    assert audit["legacy_weight_aliases"] == []
    assert audit["transformers_provenance"] == {
        "repository": "https://github.com/rwkv-rs/transformers-rwkv.git",
        "revision": "2696927df9363b5fa175076bb827ba4da2c4e581",
        "installation_source": "editable-git",
        "editable": True,
    }
    assert not (tmp_path / "rwkv7_low_rank_w8.safetensors").exists()
    serialized_config = json.loads(
        (tmp_path / "config.json").read_text(encoding="utf-8")
    )
    assert serialized_config["quantization_config"]["quantization_status"] == (
        "compressed"
    )
    assert serialized_config["rwkv7_quantization_metadata"]["target_policy"][
        "recipe"
    ]["quantization_applied"] is True

    return tmp_path, contract, audit


@pytest.mark.integration
def test_low_rank_w8_uses_standard_linear_serialization_and_audit(
    standard_linear_w8_artifact,
):
    artifact_path, contract, audit = standard_linear_w8_artifact

    assert artifact_path.joinpath("model.safetensors").is_file()
    assert len(contract.vllm.quantized_modules) == 16
    assert len(contract.vllm.quantized_low_rank_modules) == 12
    assert audit["standard_linear_ownership"] is True
    assert audit["legacy_weight_aliases"] == []


@pytest.mark.integration
def test_compressed_artifact_rejects_unapplied_recipe_metadata(
    standard_linear_w8_artifact,
):
    artifact_path, contract, _ = standard_linear_w8_artifact
    config_path = artifact_path / "config.json"
    serialized_config = json.loads(config_path.read_text(encoding="utf-8"))
    serialized_contract = serialized_config["rwkv7_quantization_metadata"]
    serialized_contract["target_policy"]["recipe"][
        "quantization_applied"
    ] = False

    with pytest.raises(ValueError, match="must record applied quantization"):
        RWKV7ArtifactContract.model_validate(serialized_contract)

    config_path.write_text(
        json.dumps(serialized_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="artifact metadata is invalid"):
        audit_rwkv7_quantized_checkpoint(
            artifact_path,
            contract.vllm.quantized_modules,
            "w8a16-low-rank-critical-high",
            contract,
        )


@pytest.mark.integration
def test_standard_rwkv7_unquantized_fresh_process_forward_and_generate(tmp_path):
    model = _tiny_standard_rwkv7().eval()
    model.save_pretrained(tmp_path)

    fresh_process = _run_fresh_process_load(tmp_path, compressed=False)

    assert fresh_process.returncode == 0, fresh_process.stderr
    evidence = json.loads(fresh_process.stdout.strip().splitlines()[-1])
    assert evidence == {
        "logits_shape": [1, 3, 32],
        "generated_shape": [1, 5],
        "missing_quantized_weights": [],
        "quantized_low_rank_module_count": 0,
        "protected_v_first_linear_module_count": 0,
        "transformers_provenance_validated": False,
    }


@pytest.mark.integration
def test_low_rank_w8_fresh_process_public_load_forward_and_generate(
    standard_linear_w8_artifact,
):
    artifact_path, _, _ = standard_linear_w8_artifact

    fresh_process = _run_fresh_process_load(
        artifact_path,
        compressed=True,
    )
    assert fresh_process.returncode == 0, fresh_process.stderr
    evidence = json.loads(fresh_process.stdout.strip().splitlines()[-1])
    assert evidence == {
        "logits_shape": [1, 3, 32],
        "generated_shape": [1, 5],
        "missing_quantized_weights": [],
        "quantized_low_rank_module_count": 12,
        "protected_v_first_linear_module_count": 2,
        "transformers_provenance_validated": True,
    }


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
