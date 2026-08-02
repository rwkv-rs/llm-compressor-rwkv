import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from llmcompressor.modifiers.quantization.rwkv7 import quantize_rwkv7_oneshot


@pytest.fixture
def cleanup_owned_nvcc_temporary_paths():
    root = Path(tempfile.gettempdir())
    owned_pattern = re.compile(r"^(tmpxft_|cc).+|^tmp.+\.build-temp$")

    def owned_paths():
        paths = set()
        for path in root.iterdir():
            if not owned_pattern.fullmatch(path.name):
                continue
            paths.add(path)
            if path.is_dir():
                paths.update(path.rglob("*"))
        return paths

    before = owned_paths()
    yield
    created = owned_paths() - before
    for path in sorted(created, key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            if not any(path.iterdir()):
                path.rmdir()
        else:
            path.unlink()


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
        lambda *args: {"format": "nvfp4-pack-quantized"},
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="reload blocked"),
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
    assert metadata["failures"][0]["error"] == "W4A4 unavailable"


@pytest.mark.skipif(
    os.environ.get("LLMCOMPRESSOR_RWKV7_GPU_TEST") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 12,
    reason="requires opt-in Blackwell NVFP4 execution",
)
@pytest.mark.parametrize("candidate", ["nvfp4-w4a4", "nvfp4-w4a16"])
@pytest.mark.integration
def test_gb10_real_nvfp4_checkpoint_has_packed_tensors_and_forward(
    tmp_path, cleanup_owned_nvcc_temporary_paths, candidate
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
    assert metadata["fresh_reload"]["passed"] is False
    assert metadata["fresh_reload"]["source_owner"] == "Transformers RWKV7 loader"
    assert (
        "must bypass Rwkv7PreTrainedModel._init_weights"
        in metadata["fresh_reload"]["regression_expectation"]
    )
    assert (
        "'Linear' object has no attribute 'weight'"
        in metadata["fresh_reload"]["stderr"]
    )
