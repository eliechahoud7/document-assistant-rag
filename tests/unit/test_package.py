import importlib
import socket

import pytest

import doc_assistant

SUBPACKAGES = [
    "domain",
    "ingestion",
    "indexing",
    "retrieval",
    "generation",
    "services",
    "api",
]


def test_version_is_defined() -> None:
    assert doc_assistant.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_are_importable(name: str) -> None:
    module = importlib.import_module(f"doc_assistant.{name}")
    assert module.__doc__


def test_network_is_blocked_by_default() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(RuntimeError):
        sock.connect(("example.com", 80))
