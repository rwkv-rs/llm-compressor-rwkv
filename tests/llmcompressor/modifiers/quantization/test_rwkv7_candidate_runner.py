"""Contracts for formal RWKV-7 checkpoint candidate execution."""

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import llmcompressor.modifiers.quantization.rwkv7 as rwkv7_module
from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.quantization.rwkv7 import (
    RWKV7ArtifactContract,
    RWKV7CheckpointContract,
    RWKV7ImplementationProvenance,
    RWKV7TransformersProvenance,
    _artifact_file_manifest,
    _fresh_reload_generate_script,
    _load_calibration_records,
    _prepare_standard_rwkv7_checkpoint,
    _snapshot_rwkv7_protected_parameters,
    _validate_native_rwkv7_config,
    _validate_native_rwkv7_runtime,
    _validate_rwkv7_loading_info,
    _verify_rwkv7_protected_parameters,
    audit_rwkv7_quantized_checkpoint,
    build_rwkv7_artifact_contract,
    build_rwkv7_quantization_recipe,
    run_rwkv7_checkpoint_candidate,
    validate_rwkv7_implementation_provenance,
    validate_rwkv7_transformers_provenance,
    verify_rwkv7_checkpoint,
)


def _operator_runtime_provenance():
    return {
        "distribution": "flash-linear-attention",
        "distribution_version": "0.5.2",
        "extra": "flash-rwkv",
        "flash_rwkv_distribution": "flash-rwkv",
        "flash_rwkv_distribution_version": "0.1.0",
        "flash_rwkv_repository": "https://github.com/rwkv-rs/FlashRWKV.git",
        "flash_rwkv_revision": "866aafd2eed146b0eda1ce03444009ae030f89e3",
        "flash_rwkv_source_kind": "vcs",
        "repository": "https://github.com/rwkv-rs/fla-rwkv.git",
        "requirement": (
            "flash-linear-attention[flash-rwkv] @ "
            "git+https://github.com/rwkv-rs/fla-rwkv.git@"
            "a4a8aa98df6ec5322f194a80ec57363dd045adfc"
        ),
        "revision": "a4a8aa98df6ec5322f194a80ec57363dd045adfc",
        "source_kind": "vcs",
    }


def _transformers_provenance():
    return RWKV7TransformersProvenance(
        repository="https://github.com/rwkv-rs/transformers-rwkv.git",
        revision="2696927df9363b5fa175076bb827ba4da2c4e581",
        installation_source="editable-git",
        editable=True,
    )


def _runtime_provenance(*_args):
    return _transformers_provenance().model_copy(
        update={"operator_runtime": _operator_runtime_provenance()}
    )


def _implementation_provenance(revision="f" * 40):
    return RWKV7ImplementationProvenance(
        repository="https://github.com/rwkv-rs/llm-compressor-rwkv.git",
        revision=revision,
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
    RWKV7ArtifactContract,
    RWKV7RepositoryContract,
    validate_rwkv7_transformers_provenance,
)
"""
        provenance_check = """
with open(f'{sys.argv[1]}/config.json', encoding='utf-8') as config_handle:
    serialized_config = json.load(config_handle)
contract = serialized_config['rwkv7_quantization_metadata']
contract = RWKV7ArtifactContract.model_validate(contract).model_dump(mode='json')
validate_rwkv7_transformers_provenance(
    RWKV7RepositoryContract.model_validate(contract['repository'])
)
transformers_provenance_validated = True
"""
        quantization_import = (
            "from transformers.utils.quantization_config import CompressedTensorsConfig"
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
missing_protected = sorted(
    set(contract['vllm']['protected_parameter_keys'])
    & set(loading_info['missing_keys'])
)
assert not missing_protected, missing_protected
quantized_low_rank = [
    model.get_submodule(name)
    for name in contract['vllm']['quantized_low_rank_modules']
]
protected_v_first = [
    model.get_submodule(name)
    for name in contract['vllm']['protected_v_first_linear_modules']
]
protected_parameters = [
    model.get_parameter(name)
    for name in contract['vllm']['protected_parameter_keys']
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
protected_parameter_count = len(protected_parameters)
"""
    else:
        contract_checks = """
missing = []
quantized_low_rank_count = 0
protected_v_first_count = 0
protected_parameter_count = 0
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
    'protected_parameter_count': protected_parameter_count,
    'transformers_provenance_validated': transformers_provenance_validated,
}}))
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(model_path)],
        capture_output=True,
        text=True,
    )


@pytest.mark.unit
def test_rwkv7_transformers_provenance_delegates_operator_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rwkv7_module,
        "_installed_transformers_provenance",
        _transformers_provenance,
    )
    monkeypatch.setattr(
        "transformers.models.rwkv7.validate_rwkv7_runtime_provenance",
        lambda: calls.append("called") or _operator_runtime_provenance(),
    )

    provenance = validate_rwkv7_transformers_provenance()

    assert calls == ["called"]
    assert provenance.repository == ("https://github.com/rwkv-rs/transformers-rwkv.git")
    assert provenance.revision == ("2696927df9363b5fa175076bb827ba4da2c4e581")
    assert provenance.installation_source == "editable-git"
    assert provenance.editable is True
    assert provenance.operator_runtime == dict(
        sorted(_operator_runtime_provenance().items())
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "repository",
    [
        "https://github.com/rwkv-rs/transformers-rwkv",
        "https://github.com/rwkv-rs/transformers-rwkv.git",
        "https://github.com/rwkv-rs/transformers-rwkv/",
        "git+https://github.com/rwkv-rs/transformers-rwkv.git/",
        "HTTPS://GITHUB.COM/RWKV-RS/TRANSFORMERS-RWKV.GIT/",
    ],
    ids=["plain", "git-suffix", "trailing-slash", "git-plus", "url-case"],
)
def test_rwkv7_repository_canonicalizer_accepts_exact_https_variants(repository):
    assert rwkv7_module._canonical_repository_url(repository) == (
        "https://github.com/rwkv-rs/transformers-rwkv"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "repository",
    [
        " https://github.com/rwkv-rs/transformers-rwkv.git",
        "https://github.com/rwkv-rs/transformers-rwkv.git ",
        "https://github.com/rwkv-rs/transformers-rwkv.git\n",
        "https://github.com/rwkv-rs/transformers-rwkv.git\x1f",
        "https://github.com/rwkv-rs/transformers-rwkv.git\x7f",
        "https://githu\N{CYRILLIC SMALL LETTER VE}.com/rwkv-rs/transformers-rwkv.git",
        "https://github.com/%72wkv-rs/transformers-rwkv.git",
        "http://github.com/rwkv-rs/transformers-rwkv.git",
        "ssh://git@github.com/rwkv-rs/transformers-rwkv.git",
        "https://user@github.com/rwkv-rs/transformers-rwkv.git",
        "https://github.com:443/rwkv-rs/transformers-rwkv.git",
        "https://github.com/rwkv-rs/transformers-rwkv.git?ref=main",
        "https://github.com/rwkv-rs/transformers-rwkv.git#main",
        "https://github.example/rwkv-rs/transformers-rwkv.git",
        "https://github.com/attacker/transformers-rwkv.git",
        "https://github.com/rwkv-rs//transformers-rwkv.git",
        "https://github.com/rwkv-rs/transformers-rwkv.git.git",
        "https://github.com/rwkv-rs/transformers-rwkv.git//",
        "https://github.com/rwkv-rs/transformers-rwkv.git;ref=main",
    ],
    ids=[
        "leading-space",
        "trailing-space",
        "trailing-newline",
        "control",
        "delete",
        "unicode-host",
        "percent-encoding",
        "http",
        "ssh",
        "userinfo",
        "port",
        "query",
        "fragment",
        "foreign-host",
        "fork",
        "repeated-path-slash",
        "double-git-suffix",
        "double-trailing-slash",
        "parameters",
    ],
)
def test_rwkv7_transformers_pep610_rejects_hostile_repository(
    monkeypatch,
    repository,
):
    import transformers

    class _VcsDistribution:
        @staticmethod
        def read_text(filename):
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": repository,
                    "vcs_info": {
                        "vcs": "git",
                        "requested_revision": (
                            "2696927df9363b5fa175076bb827ba4da2c4e581"
                        ),
                        "commit_id": "2696927df9363b5fa175076bb827ba4da2c4e581",
                    },
                }
            )

        @staticmethod
        def locate_file(filename):
            assert filename == "transformers/__init__.py"
            return transformers.__file__

    monkeypatch.setattr(
        rwkv7_module.importlib_metadata,
        "distribution",
        lambda name: _VcsDistribution(),
    )

    with pytest.raises(RuntimeError, match="RWKV-7 repository URL"):
        validate_rwkv7_transformers_provenance()


@pytest.mark.unit
def test_rwkv7_transformers_provenance_propagates_operator_failure(monkeypatch):
    monkeypatch.setattr(
        rwkv7_module,
        "_installed_transformers_provenance",
        _transformers_provenance,
    )

    def fail_operator_provenance():
        raise RuntimeError("operator provenance unavailable")

    monkeypatch.setattr(
        "transformers.models.rwkv7.validate_rwkv7_runtime_provenance",
        fail_operator_provenance,
    )

    with pytest.raises(RuntimeError, match="operator provenance unavailable"):
        validate_rwkv7_transformers_provenance()


@pytest.mark.unit
def test_rwkv7_editable_provenance_rejects_repo_local_shadow_module(
    tmp_path, monkeypatch
):
    import transformers

    repository_root = tmp_path / "transformers-rwkv"
    shadow_module = (
        repository_root / ".venv/lib/python3.12/site-packages/transformers/__init__.py"
    )
    shadow_module.parent.mkdir(parents=True)
    shadow_module.write_text("", encoding="utf-8")

    class _EditableDistribution:
        @staticmethod
        def read_text(filename):
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": repository_root.as_uri(),
                    "dir_info": {"editable": True},
                }
            )

    monkeypatch.setattr(
        rwkv7_module.importlib_metadata,
        "distribution",
        lambda name: _EditableDistribution(),
    )
    monkeypatch.setattr(transformers, "__file__", str(shadow_module))
    monkeypatch.setattr(
        rwkv7_module,
        "_git_provenance_value",
        lambda source, *arguments: str(repository_root),
    )

    with pytest.raises(RuntimeError, match="does not belong to the editable"):
        rwkv7_module._installed_transformers_provenance()


@pytest.mark.unit
def test_rwkv7_implementation_provenance_binds_clean_editable_checkout(
    tmp_path, monkeypatch
):
    import llmcompressor

    repository_root = tmp_path / "llm-compressor-rwkv"
    module_path = repository_root / "src/llmcompressor/__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("", encoding="utf-8")
    revision = "f" * 40

    class _EditableDistribution:
        metadata = {"Name": "llmcompressor"}

        @staticmethod
        def read_text(filename):
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": repository_root.as_uri(),
                    "dir_info": {"editable": True},
                }
            )

    def git_value(source, *arguments):
        assert source == repository_root
        values = {
            ("rev-parse", "--show-toplevel"): str(repository_root),
            ("remote", "get-url", "origin"): (
                "https://github.com/rwkv-rs/llm-compressor-rwkv.git"
            ),
            ("rev-parse", "HEAD"): revision,
        }
        return values[arguments]

    monkeypatch.setattr(
        rwkv7_module.importlib_metadata,
        "distribution",
        lambda name: _EditableDistribution(),
    )
    monkeypatch.setattr(llmcompressor, "__file__", str(module_path))
    monkeypatch.setattr(rwkv7_module, "_git_provenance_value", git_value)
    monkeypatch.setattr(
        rwkv7_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
    )

    provenance = validate_rwkv7_implementation_provenance(revision)

    assert provenance == _implementation_provenance(revision)
    with pytest.raises(RuntimeError, match="differs from the active checkout"):
        validate_rwkv7_implementation_provenance("e" * 40)


@pytest.mark.unit
def test_formal_runner_fails_provenance_before_output_mutation(tmp_path, monkeypatch):
    output_dir = tmp_path / "result"
    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_transformers_provenance",
        _runtime_provenance,
    )

    def reject_implementation(*args, **kwargs):
        raise RuntimeError("implementation provenance unavailable")

    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_implementation_provenance",
        reject_implementation,
    )

    with pytest.raises(RuntimeError, match="implementation provenance unavailable"):
        run_rwkv7_checkpoint_candidate(
            tmp_path / "checkpoint.pth",
            tmp_path / "calibration.jsonl",
            output_dir,
            calibration_sha256="0" * 64,
            implementation_revision="f" * 40,
            candidate="nvfp4-w4a4",
        )
    assert not output_dir.exists()


@pytest.mark.unit
def test_standard_checkpoint_reuse_binds_converter_provenance_and_manifest(
    tmp_path, monkeypatch
):
    import transformers

    destination = tmp_path / "baseline-standard-hf"
    destination.mkdir()
    (destination / "config.json").write_text("{}\n", encoding="utf-8")
    (destination / "model.safetensors").write_bytes(b"synthetic")
    checkpoint_contract = RWKV7CheckpointContract()
    runtime_provenance = _runtime_provenance()
    implementation_provenance = _implementation_provenance()
    provenance = {
        "schema_version": 2,
        "checkpoint": checkpoint_contract.model_dump(mode="json"),
        "converter_runtime": runtime_provenance.model_dump(mode="json"),
        "implementation": implementation_provenance.model_dump(mode="json"),
        "artifact_manifest": _artifact_file_manifest(destination),
    }
    provenance_path = destination / "rwkv7_source_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="rwkv7",
            architectures=["Rwkv7ForCausalLM"],
            embedding_layer_norm_fused=False,
        ),
    )

    manifest = _prepare_standard_rwkv7_checkpoint(
        tmp_path / checkpoint_contract.filename,
        destination,
        checkpoint_contract,
        runtime_provenance,
        implementation_provenance,
    )
    assert manifest["file_count"] == 3

    (destination / "config.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance or manifest drifted"):
        _prepare_standard_rwkv7_checkpoint(
            tmp_path / checkpoint_contract.filename,
            destination,
            checkpoint_contract,
            runtime_provenance,
            implementation_provenance,
        )


@pytest.mark.unit
def test_rwkv7_transformers_provenance_rejects_registry_install(monkeypatch):
    class _RegistryDistribution:
        @staticmethod
        def read_text(filename):
            assert filename == "direct_url.json"
            return None

    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7.importlib_metadata.distribution",
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
                        "commit_id": ("2696927df9363b5fa175076bb827ba4da2c4e581"),
                    },
                }
            )

        @staticmethod
        def locate_file(filename):
            assert filename == "transformers/__init__.py"
            return transformers.__file__

    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7.importlib_metadata.distribution",
        lambda name: _BranchVcsDistribution(),
    )

    with pytest.raises(RuntimeError, match="requested revision is not exact"):
        validate_rwkv7_transformers_provenance()


@pytest.mark.unit
def test_rwkv7_transformers_provenance_rejects_revision_drift(monkeypatch):
    monkeypatch.setattr(
        "llmcompressor.modifiers.quantization.rwkv7._installed_transformers_provenance",
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
def test_artifact_contract_pins_fork_standard_names_and_v_first_protection(
    monkeypatch,
):
    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_transformers_provenance",
        _runtime_provenance,
    )
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
    assert contract.runtime_provenance == _runtime_provenance()
    assert (
        contract.target_policy.recipe.runtime_provenance == contract.runtime_provenance
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
    assert contract.vllm.consumer_capabilities == [
        "transformers-rwkv-compressed-tensors"
    ]
    assert contract.vllm.vllm_consumer_requirement == ("vllm-rwkv-nvfp4-w4a4")
    assert contract.vllm.vllm_consumer_revision is None
    assert contract.vllm.target_schema_version == 1
    assert contract.vllm.target_schema == "rwkv7-nvfp4-critical-high-v1"
    assert contract.vllm.num_hidden_layers == 2
    assert (
        contract.vllm.quantized_target_fqns_digest
        == hashlib.sha256(
            json.dumps(
                contract.vllm.quantized_modules,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert contract.vllm.layer_zero_v_first_producer == ("model.blocks.0.att.value")
    assert contract.vllm.protected_embedding_modules == ["model.embeddings"]
    assert "head" in contract.vllm.protected_linear_modules
    assert "model.blocks.0.ln0" in contract.vllm.protected_normalization_modules
    assert "model.blocks.1.ln0" not in (contract.vllm.protected_normalization_modules)
    assert "model.blocks.0.att.value" in contract.vllm.protected_modules
    assert "model.blocks.1.att.v0" in contract.vllm.protected_tensors
    assert "model.blocks.0.ffn.x_k" in contract.vllm.protected_state_tensors
    assert {
        "model.blocks.0.ln1.bias",
        "model.blocks.0.att.ln_x.bias",
    } <= set(contract.vllm.protected_parameter_keys)
    assert contract.vllm.quantized_modules == [
        "model.blocks.0.ffn.key",
        "model.blocks.0.ffn.value",
        "model.blocks.1.ffn.key",
        "model.blocks.1.ffn.value",
    ]
    assert contract.formal_checkpoint is True
    assert contract.formal_evaluation is False
    assert contract.target_policy.recipe.quantization_applied is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("candidate", "target_schema"),
    [
        ("nvfp4-w4a16", "rwkv7-nvfp4-critical-high-v1"),
        (
            "nvfp4-w4a16-protection-ablation",
            "rwkv7-nvfp4-protection-ablation-no-ffn-v1",
        ),
    ],
)
def test_w4a16_artifact_declares_exact_vllm_capability_and_target_schema(
    monkeypatch,
    candidate,
    target_schema,
):
    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_transformers_provenance",
        _runtime_provenance,
    )
    modifier = build_rwkv7_quantization_recipe(_tiny_standard_rwkv7(), candidate)
    contract = build_rwkv7_artifact_contract(
        modifier.target_policy_metadata,
        candidate,
    )

    assert contract.vllm.consumer_capabilities == [
        "transformers-rwkv-compressed-tensors",
        "vllm-rwkv-nvfp4-w4a16",
    ]
    assert contract.vllm.vllm_consumer_requirement == ("vllm-rwkv-nvfp4-w4a16")
    assert contract.vllm.vllm_consumer_revision == (
        "88b992bbc73e8b904ae672dfd39396b6dd0d6ea4"
    )
    assert contract.vllm.target_schema == target_schema
    if candidate == "nvfp4-w4a16-protection-ablation":
        assert len(contract.vllm.quantized_modules) == 9 + 12 * (2 - 1)
        assert not any(".ffn." in name for name in contract.vllm.quantized_modules)
        assert contract.vllm.quantized_modules[8] == ("model.blocks.0.att.output")
        assert contract.vllm.quantized_modules[9] == "model.blocks.1.att.w1"


@pytest.mark.unit
def test_protected_snapshot_rejects_parameter_value_rewrite(monkeypatch):
    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_transformers_provenance",
        _runtime_provenance,
    )
    model = _tiny_standard_rwkv7().eval()
    modifier = build_rwkv7_quantization_recipe(model, "nvfp4-w4a16")
    contract = build_rwkv7_artifact_contract(
        modifier.target_policy_metadata,
        "nvfp4-w4a16",
    )
    snapshot = _snapshot_rwkv7_protected_parameters(model, contract)
    layer_norm = model.model.blocks[0].ln1
    with torch.no_grad():
        layer_norm.bias.add_(1)

    with pytest.raises(RuntimeError, match="changed a protected tensor value"):
        _verify_rwkv7_protected_parameters(model, contract, snapshot)


@pytest.mark.integration
@pytest.mark.parametrize(
    "candidate",
    ["nvfp4-w4a16", "nvfp4-w4a16-protection-ablation"],
    ids=["ordinary", "protection-ablation"],
)
def test_w4a16_real_serializer_preserves_mainstream_config_contract(
    tmp_path,
    monkeypatch,
    candidate,
):
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )

    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_transformers_provenance",
        _runtime_provenance,
    )
    model = _tiny_standard_rwkv7().eval()
    modifier = build_rwkv7_quantization_recipe(model, candidate)
    contract = build_rwkv7_artifact_contract(
        modifier.target_policy_metadata,
        candidate,
    )
    protected_snapshot = _snapshot_rwkv7_protected_parameters(model, contract)
    resolved_groups = modifier.resolved_config.config_groups
    assert list(resolved_groups) == ["group_0"]
    assert resolved_groups["group_0"].targets == ["Linear"]

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
    protection_audit = _verify_rwkv7_protected_parameters(
        model,
        contract,
        protected_snapshot,
    )
    assert protection_audit["passed"] is True
    assert protection_audit["module_identity_preserved"] is True
    assert protection_audit["parameter_ownership_preserved"] is True
    assert protection_audit["parameter_values_preserved"] is True
    assert protection_audit["parameter_count"] == len(
        contract.vllm.protected_parameter_keys
    )
    assert all(
        getattr(model.get_submodule(name), "quantization_scheme", None) is not None
        for name in contract.vllm.quantized_modules
    )

    artifact_path = tmp_path / candidate
    setattr(
        model.config,
        "rwkv7_quantization_metadata",
        contract.model_dump(mode="json"),
    )
    modify_save_pretrained(model)
    model.save_pretrained(artifact_path, save_compressed=True)
    serialized = json.loads(
        (artifact_path / "config.json").read_text(encoding="utf-8")
    )["quantization_config"]
    assert list(serialized["config_groups"]) == ["group_0"]
    assert serialized["config_groups"]["group_0"]["targets"] == ["Linear"]
    assert set(serialized["ignore"]) == set(contract.vllm.protected_linear_modules)

    audit = audit_rwkv7_quantized_checkpoint(
        artifact_path,
        contract.vllm.quantized_modules,
        candidate,
        contract,
        expected_protected_parameter_sha256=protection_audit["parameter_sha256"],
    )
    assert audit["targets"] == contract.vllm.quantized_modules
    assert audit["tensor_count"] == audit["expected_tensor_count"]
    assert audit["legacy_weight_aliases"] == []
    assert audit["protected_parameter_values_verified"] is True

    if candidate == "nvfp4-w4a16":
        fresh_process = subprocess.run(
            [
                sys.executable,
                "-c",
                _fresh_reload_generate_script(),
                str(artifact_path),
                "20260801",
                "[1, 2, 3, 4]",
                "1",
                "1",
                "1",
                "load-only",
                "cpu",
                json.dumps(protection_audit["parameter_sha256"], sort_keys=True),
            ],
            capture_output=True,
            text=True,
        )
        assert fresh_process.returncode == 0, fresh_process.stderr
        load_evidence = json.loads(fresh_process.stdout.strip().splitlines()[-1])
        assert load_evidence["execution_mode"] == "load-only"
        assert load_evidence["artifact_contract_validated"] is True
        assert load_evidence["transformers_provenance"] is None
        assert load_evidence["standard_generate"] == {
            "passed": False,
            "executed": False,
        }
        strict_load = load_evidence["standard_linear_load"]
        assert strict_load["loader"] == "Rwkv7ForCausalLM.from_pretrained"
        assert strict_load["config_class"] == "Rwkv7Config"
        assert strict_load["model_class"] == "Rwkv7ForCausalLM"
        assert strict_load["trust_remote_code"] is False
        assert strict_load["strict_loading_info"] is True
        assert strict_load["runtime_float_weights_restored"] is True
        assert strict_load["protected_parameter_values_verified"] is True
        assert strict_load["vllm_metadata_validated"] is True
        assert strict_load["quantized_scheme_count"] == len(
            contract.vllm.quantized_modules
        )
        assert strict_load["protected_parameter_count"] == len(
            contract.vllm.protected_parameter_keys
        )

        from safetensors.torch import load_file, save_file

        protected_name = "model.blocks.0.ln1.bias"
        shard = artifact_path / "model.safetensors"
        tensors = load_file(shard)
        tensors[protected_name] = tensors[protected_name].clone()
        tensors[protected_name].view(-1)[0] += 1
        save_file(tensors, shard)
        with pytest.raises(RuntimeError, match="protected tensor values drifted"):
            audit_rwkv7_quantized_checkpoint(
                artifact_path,
                contract.vllm.quantized_modules,
                candidate,
                contract,
                expected_protected_parameter_sha256=(
                    protection_audit["parameter_sha256"]
                ),
            )


@pytest.fixture
def standard_linear_w8_artifact(tmp_path, monkeypatch):
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )

    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_transformers_provenance",
        _runtime_provenance,
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
    assert contract.vllm.consumer_capabilities == [
        "transformers-rwkv-compressed-tensors"
    ]
    assert contract.vllm.vllm_consumer_requirement is None
    assert contract.vllm.vllm_consumer_revision is None
    assert contract.vllm.target_schema == ("rwkv7-w8-low-rank-critical-high-v1")
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
    assert contract.vllm.protected_v_first_linear_modules == [
        "model.blocks.1.att.v1",
        "model.blocks.1.att.v2",
    ]
    assert contract.vllm.low_rank_linear_modules == sorted(
        [*low_rank_modules, *contract.vllm.protected_v_first_linear_modules]
    )
    assert contract.vllm.quantized_weight_names == [
        f"{name}.weight" for name in contract.vllm.quantized_modules
    ]
    fresh_reload_script = _fresh_reload_generate_script()
    assert "rwkv7_low_rank_w8" not in fresh_reload_script
    assert "output_loading_info=True" in fresh_reload_script
    assert "_validate_rwkv7_loading_info(loading_info)" in fresh_reload_script
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
    assert audit["tensor_count"] == audit["expected_tensor_count"]
    assert audit["transformers_provenance"] == _runtime_provenance().model_dump(
        mode="json"
    )
    assert not (tmp_path / "rwkv7_low_rank_w8.safetensors").exists()
    serialized_config = json.loads(
        (tmp_path / "config.json").read_text(encoding="utf-8")
    )
    assert serialized_config["quantization_config"]["quantization_status"] == (
        "compressed"
    )
    assert (
        serialized_config["rwkv7_quantization_metadata"]["target_policy"]["recipe"][
            "quantization_applied"
        ]
        is True
    )

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
def test_audit_uses_complete_serialized_contract_without_external_copy(
    standard_linear_w8_artifact,
):
    artifact_path, contract, _ = standard_linear_w8_artifact

    audit = audit_rwkv7_quantized_checkpoint(
        artifact_path,
        contract.vllm.quantized_modules,
        contract.candidate,
    )

    assert audit["artifact_contract_serialized"] is True
    assert audit["protected_parameter_count"] == len(
        contract.vllm.protected_parameter_keys
    )


@pytest.mark.integration
def test_audit_rejects_partial_target_inventory(standard_linear_w8_artifact):
    artifact_path, contract, _ = standard_linear_w8_artifact

    with pytest.raises(RuntimeError, match="complete serialized inventory"):
        audit_rwkv7_quantized_checkpoint(
            artifact_path,
            contract.vllm.quantized_modules[:-1],
            contract.candidate,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "parameter_name",
    [
        "model.blocks.0.ln1.bias",
        "model.blocks.0.att.ln_x.bias",
    ],
    ids=["layer-norm-bias", "group-norm-bias"],
)
def test_audit_rejects_missing_protected_normalization_bias_before_runtime(
    standard_linear_w8_artifact,
    parameter_name,
):
    from safetensors.torch import load_file, save_file

    artifact_path, contract, _ = standard_linear_w8_artifact
    assert parameter_name in contract.vllm.protected_parameter_keys
    shard = artifact_path / "model.safetensors"
    tensors = load_file(shard)
    tensors.pop(parameter_name)
    save_file(tensors, shard)

    with pytest.raises(RuntimeError, match="missing protected physical tensors"):
        audit_rwkv7_quantized_checkpoint(
            artifact_path,
            contract.vllm.quantized_modules,
            contract.candidate,
        )


@pytest.mark.integration
def test_audit_rejects_unexpected_physical_tensor(
    standard_linear_w8_artifact,
):
    from safetensors.torch import load_file, save_file

    artifact_path, contract, _ = standard_linear_w8_artifact
    shard = artifact_path / "model.safetensors"
    tensors = load_file(shard)
    tensors["model.unowned_extra"] = torch.zeros(1)
    save_file(tensors, shard)

    with pytest.raises(RuntimeError, match="exact artifact contract"):
        audit_rwkv7_quantized_checkpoint(
            artifact_path,
            contract.vllm.quantized_modules,
            contract.candidate,
        )


@pytest.mark.integration
def test_audit_rejects_legacy_logical_weight_alias(
    standard_linear_w8_artifact,
):
    from safetensors.torch import load_file, save_file

    artifact_path, contract, _ = standard_linear_w8_artifact
    shard = artifact_path / "model.safetensors"
    tensors = load_file(shard)
    target = contract.vllm.quantized_modules[0]
    tensors[f"{target}.weight"] = torch.zeros(1)
    save_file(tensors, shard)

    with pytest.raises(RuntimeError, match="legacy raw weight aliases"):
        audit_rwkv7_quantized_checkpoint(
            artifact_path,
            contract.vllm.quantized_modules,
            contract.candidate,
        )


@pytest.mark.integration
def test_audit_rejects_missing_packed_tensor(standard_linear_w8_artifact):
    from safetensors.torch import load_file, save_file

    artifact_path, contract, _ = standard_linear_w8_artifact
    shard = artifact_path / "model.safetensors"
    tensors = load_file(shard)
    tensors.pop(f"{contract.vllm.quantized_modules[0]}.weight_scale")
    save_file(tensors, shard)

    with pytest.raises(RuntimeError, match="compressed tensor inventory drifted"):
        audit_rwkv7_quantized_checkpoint(
            artifact_path,
            contract.vllm.quantized_modules,
            contract.candidate,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("weights", "symmetric"), False),
        (("weights", "num_bits"), 4),
        (("input_activations",), {"dynamic": "local"}),
    ],
    ids=["symmetric", "bits", "unexpected-input-args"],
)
def test_audit_rejects_serialized_config_group_drift(
    standard_linear_w8_artifact,
    path,
    replacement,
):
    artifact_path, contract, _ = standard_linear_w8_artifact
    config_path = artifact_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = config["quantization_config"]["config_groups"]["group_0"]
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="quantization config group differs"):
        audit_rwkv7_quantized_checkpoint(
            artifact_path,
            contract.vllm.quantized_modules,
            contract.candidate,
        )


@pytest.mark.integration
def test_audit_accepts_serialized_ignore_order_variation(
    standard_linear_w8_artifact,
):
    artifact_path, contract, _ = standard_linear_w8_artifact
    config_path = artifact_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantization_config"]["ignore"].reverse()
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit = audit_rwkv7_quantized_checkpoint(
        artifact_path,
        contract.vllm.quantized_modules,
        contract.candidate,
    )
    assert audit["targets"] == contract.vllm.quantized_modules


@pytest.mark.integration
@pytest.mark.parametrize(
    "drift",
    [
        "missing-ignore",
        "extra-ignore",
        "duplicate-ignore",
        "empty-ignore",
        "empty-string-ignore",
        "non-string-ignore",
        "expanded-targets",
    ],
)
def test_audit_rejects_serialized_producer_contract_drift(
    standard_linear_w8_artifact,
    drift,
):
    artifact_path, contract, _ = standard_linear_w8_artifact
    config_path = artifact_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config["quantization_config"]
    if drift == "missing-ignore":
        quantization["ignore"].pop()
        error_match = "protected Linear ignore inventory"
    elif drift == "extra-ignore":
        quantization["ignore"].append("model.blocks.0.unowned")
        error_match = "protected Linear ignore inventory"
    elif drift == "duplicate-ignore":
        quantization["ignore"].append(quantization["ignore"][0])
        error_match = "must not contain duplicates"
    elif drift == "empty-ignore":
        quantization["ignore"] = []
        error_match = "must be a non-empty list"
    elif drift == "empty-string-ignore":
        quantization["ignore"].append("")
        error_match = "entries must be non-empty"
    elif drift == "non-string-ignore":
        quantization["ignore"].append(7)
        error_match = "entries must be non-empty"
    else:
        quantization["config_groups"]["group_0"]["targets"] = (
            contract.vllm.quantized_modules
        )
        error_match = "must target exactly"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=error_match):
        audit_rwkv7_quantized_checkpoint(
            artifact_path,
            contract.vllm.quantized_modules,
            contract.candidate,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("field", "replacement", "error_match"),
    [
        ("model_type", "rwkv", "model_type"),
        ("architectures", ["AutoModelForCausalLM"], "architectures"),
        (
            "auto_map",
            {"AutoConfig": "configuration_rwkv7.Rwkv7Config"},
            "without auto_map",
        ),
    ],
    ids=["model-type", "architectures", "auto-map"],
)
def test_audit_rejects_native_config_schema_drift(
    standard_linear_w8_artifact,
    field,
    replacement,
    error_match,
):
    artifact_path, contract, _ = standard_linear_w8_artifact
    config_path = artifact_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = replacement
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=error_match):
        audit_rwkv7_quantized_checkpoint(
            artifact_path,
            contract.vllm.quantized_modules,
            contract.candidate,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("missing_keys", ["model.blocks.0.ln1.bias"]),
        ("unexpected_keys", ["model.unexpected.weight"]),
        (
            "mismatched_keys",
            [("model.blocks.0.ln1.weight", [8], [16])],
        ),
    ],
    ids=["protected-missing", "unexpected", "mismatched"],
)
def test_public_loader_rejects_all_state_dict_key_drift(field, value):
    loading_info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    _validate_rwkv7_loading_info(loading_info)
    loading_info[field] = value

    with pytest.raises(RuntimeError, match=f"non-empty {field}"):
        _validate_rwkv7_loading_info(loading_info)


@pytest.mark.unit
def test_native_runtime_rejects_non_native_config_or_model():
    model = _tiny_standard_rwkv7()
    serialized_config = model.config.to_dict()
    serialized_config["architectures"] = ["Rwkv7ForCausalLM"]
    _validate_native_rwkv7_config(serialized_config)
    _validate_native_rwkv7_runtime(model.config, model)

    with pytest.raises(RuntimeError, match="native Transformers Rwkv7Config"):
        _validate_native_rwkv7_runtime(SimpleNamespace(), model)
    with pytest.raises(RuntimeError, match="native Transformers Rwkv7ForCausalLM"):
        _validate_native_rwkv7_runtime(model.config, torch.nn.Linear(1, 1))


@pytest.mark.unit
def test_artifact_contract_rejects_incomplete_protected_parameter_inventory(
    monkeypatch,
):
    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_transformers_provenance",
        _runtime_provenance,
    )
    modifier = build_rwkv7_quantization_recipe(_tiny_standard_rwkv7())
    contract = build_rwkv7_artifact_contract(
        modifier.target_policy_metadata,
        "nvfp4-w4a4",
    ).model_dump(mode="json")
    contract["vllm"]["protected_parameter_keys"].remove("model.blocks.0.ln1.bias")

    with pytest.raises(ValueError, match="protected parameter inventory"):
        RWKV7ArtifactContract.model_validate(contract)


@pytest.mark.unit
def test_artifact_contract_rejects_quantized_target_digest_tamper(monkeypatch):
    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_transformers_provenance",
        _runtime_provenance,
    )
    modifier = build_rwkv7_quantization_recipe(
        _tiny_standard_rwkv7(),
        "nvfp4-w4a16-protection-ablation",
    )
    contract = build_rwkv7_artifact_contract(
        modifier.target_policy_metadata,
        "nvfp4-w4a16-protection-ablation",
    ).model_dump(mode="json")
    contract["vllm"]["quantized_target_fqns_digest"] = "0" * 64

    with pytest.raises(ValueError, match="FQN digest"):
        RWKV7ArtifactContract.model_validate(contract)


@pytest.mark.unit
def test_ablation_loader_metadata_rejects_ffn_target_injection(monkeypatch):
    monkeypatch.setattr(
        rwkv7_module,
        "validate_rwkv7_transformers_provenance",
        _runtime_provenance,
    )
    modifier = build_rwkv7_quantization_recipe(
        _tiny_standard_rwkv7(),
        "nvfp4-w4a16-protection-ablation",
    )
    loader_metadata = build_rwkv7_artifact_contract(
        modifier.target_policy_metadata,
        "nvfp4-w4a16-protection-ablation",
    ).vllm
    tampered = loader_metadata.model_dump(mode="json")
    tampered["quantized_modules"].append("model.blocks.0.ffn.key")
    tampered["quantized_weight_names"].append("model.blocks.0.ffn.key.weight")
    tampered["quantized_target_fqns_digest"] = hashlib.sha256(
        json.dumps(
            tampered["quantized_modules"],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="exact expanded target schema"):
        type(loader_metadata).model_validate(tampered)


@pytest.mark.integration
def test_compressed_artifact_rejects_unapplied_recipe_metadata(
    standard_linear_w8_artifact,
):
    artifact_path, contract, _ = standard_linear_w8_artifact
    config_path = artifact_path / "config.json"
    serialized_config = json.loads(config_path.read_text(encoding="utf-8"))
    serialized_contract = serialized_config["rwkv7_quantization_metadata"]
    serialized_contract["target_policy"]["recipe"]["quantization_applied"] = False

    with pytest.raises(ValueError, match="must record applied quantization"):
        RWKV7ArtifactContract.model_validate(serialized_contract)

    config_path.write_text(
        json.dumps(serialized_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fresh_process = subprocess.run(
        [
            sys.executable,
            "-c",
            _fresh_reload_generate_script(),
            str(artifact_path),
            "20260801",
            "[1, 2, 3, 4]",
            "1",
            "1",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert fresh_process.returncode != 0
    assert "must record applied quantization" in fresh_process.stderr
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
        "protected_parameter_count": 0,
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
        "protected_parameter_count": 53,
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
