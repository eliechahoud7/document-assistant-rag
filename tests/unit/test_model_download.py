from pathlib import Path
from typing import Any

import huggingface_hub
import pytest

from doc_assistant.config import Settings
from doc_assistant.indexing.model_download import REQUIRED_FILES, download_embedding_model


def test_downloads_only_required_files_at_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    settings = Settings(models_dir=tmp_path)

    target = download_embedding_model(settings)

    assert target == tmp_path / "multilingual-e5-small"
    assert calls == [
        {
            "repo_id": "intfloat/multilingual-e5-small",
            "revision": settings.embedding_model_revision,
            "local_dir": target,
            "allow_patterns": REQUIRED_FILES,
        }
    ]


def test_required_files_exclude_redundant_formats() -> None:
    assert "model.safetensors" in REQUIRED_FILES
    assert not any(name.endswith((".bin", ".onnx")) for name in REQUIRED_FILES)
