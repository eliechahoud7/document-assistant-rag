import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from doc_assistant.evaluation.dataset import DatasetError, load_dataset
from doc_assistant.ingestion.pdf_loader import PypdfLoader

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_V1 = ROOT / "evaluation" / "datasets" / "synthetic_v1.json"
MakePdf = Callable[..., Path]


def minimal() -> dict[str, Any]:
    return {
        "name": "mini",
        "kind": "synthetic",
        "documents": [
            {"filename": "a.pdf", "lang": "fr", "pages": ["Le chat  dort.", "Le chien court."]},
            {"filename": "b.pdf", "lang": "en", "pages": ["The cat sleeps."]},
        ],
        "questions": [
            {
                "id": "q1",
                "lang": "en",
                "category": "cross_lingual",
                "answerable": True,
                "question": "Who runs?",
                "references": [{"filename": "a.pdf", "page": 2, "quote": "chien court"}],
            },
            {
                "id": "q2",
                "lang": "fr",
                "category": "multi_passage",
                "answerable": True,
                "question": "Que fait le chat ?",
                "references": [
                    {"filename": "a.pdf", "page": 1, "quote": "chat dort"},
                    {"filename": "b.pdf", "page": 1, "quote": "cat sleeps"},
                ],
            },
            {"id": "u1", "lang": "fr", "answerable": False, "question": "Quel âge a le chat ?"},
        ],
    }


def write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_and_locates_quotes_in_cleaned_text(tmp_path: Path) -> None:
    dataset = load_dataset(write(tmp_path, minimal()))

    assert (dataset.name, dataset.kind) == ("mini", "synthetic")
    assert len(dataset.answerable) == 2 and len(dataset.unanswerable) == 1
    # "Le chat  dort." est nettoyé en "Le chat dort." : la citation y est localisée.
    reference = dataset.questions[1].references[0]
    assert dataset.documents[0].pages[0].text[reference.start : reference.end] == "chat dort"
    assert dataset.questions[2].category == "uncategorized"


def test_language_pairs(tmp_path: Path) -> None:
    dataset = load_dataset(write(tmp_path, minimal()))

    assert dataset.language_pair(dataset.questions[0]) == "en->fr"
    assert dataset.language_pair(dataset.questions[1]) == "fr->mixed"


def test_hash_identifies_the_exact_file(tmp_path: Path) -> None:
    path = write(tmp_path, minimal())
    first = load_dataset(path).sha256
    data = minimal()
    data["name"] = "mini2"

    assert load_dataset(path).sha256 == first
    assert load_dataset(write(tmp_path, data)).sha256 != first


def _set(path: list[Any], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(data: dict[str, Any]) -> None:
        target: Any = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_set(["kind"], "autre"), "kind"),
        (_set(["name"], ""), "name"),
        (_set(["documents"], []), "aucun document"),
        (_set(["questions"], []), "aucune question"),
        (_set(["documents", 1, "filename"], "a.pdf"), "double"),
        (_set(["documents", 0, "lang"], "de"), "lang"),
        (_set(["questions", 1, "id"], "q1"), "dupliqué"),
        (_set(["questions", 0, "question"], "  "), "vide"),
        (_set(["questions", 0, "lang"], "es"), "lang"),
        (_set(["questions", 0, "answerable"], "oui"), "booléen"),
        (_set(["questions", 0, "references"], []), "référence"),
        (_set(["questions", 2, "references"], [{"filename": "a.pdf"}]), "référence"),
        (_set(["questions", 0, "references", 0, "filename"], "z.pdf"), "inconnu"),
        (_set(["questions", 0, "references", 0, "page"], 3), "page"),
        (_set(["questions", 0, "references", 0, "quote"], "chien vole"), "0 fois"),
        (_set(["questions", 0, "references", 0, "quote"], ""), "vide"),
        (_set(["questions", 0, "references", 0, "quote"], "x" * 301), "300"),
        (_set(["documents", 0, "pages", 1], "Le chien court. Le chien court."), "2 fois"),
    ],
)
def test_invalid_datasets_are_rejected(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    data = minimal()
    mutation(data)

    with pytest.raises(DatasetError, match=message):
        load_dataset(write(tmp_path, data))


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text("{pas du json", encoding="utf-8")

    with pytest.raises(DatasetError, match="JSON invalide"):
        load_dataset(path)


def test_real_dataset_loads_pdfs_relative_to_the_file(tmp_path: Path, make_pdf: MakePdf) -> None:
    make_pdf([["Le chat dort sur le canapé."]], name="vrai.pdf")
    data = {
        "name": "reel",
        "kind": "real",
        "documents": [{"path": "vrai.pdf", "lang": "fr"}],
        "questions": [
            {
                "id": "q1",
                "lang": "fr",
                "answerable": True,
                "question": "Où dort le chat ?",
                "references": [{"filename": "vrai.pdf", "page": 1, "quote": "sur le canapé"}],
            }
        ],
    }

    dataset = load_dataset(write(tmp_path, data), loader=PypdfLoader())

    assert dataset.kind == "real"
    assert dataset.documents[0].filename == "vrai.pdf"


def test_real_dataset_requires_a_loader(tmp_path: Path) -> None:
    data = {"name": "reel", "kind": "real", "documents": [{"path": "x.pdf", "lang": "fr"}]}

    with pytest.raises(DatasetError, match="DocumentLoader"):
        load_dataset(write(tmp_path, data))


def test_real_template_is_well_formed_and_points_to_ignored_folder() -> None:
    template = json.loads((SYNTHETIC_V1.parent / "real_template.json").read_text("utf-8"))

    assert template["kind"] == "real"
    assert all(doc["path"].startswith("../documents/") for doc in template["documents"])
    with pytest.raises(FileNotFoundError):  # le PDF d'exemple n'existe pas : c'est un modèle
        load_dataset(SYNTHETIC_V1.parent / "real_template.json", loader=PypdfLoader())


def test_synthetic_v1_is_valid_and_balanced() -> None:
    """Intégrité du jeu versionné (citations exactes, équilibre FR/EN, catégories)."""
    dataset = load_dataset(SYNTHETIC_V1)

    assert dataset.kind == "synthetic"
    assert (len(dataset.answerable), len(dataset.unanswerable)) == (30, 10)
    assert {q.lang for q in dataset.questions} == {"fr", "en"}
    pairs = {dataset.language_pair(q) for q in dataset.answerable}
    assert {"fr->fr", "en->en", "fr->en", "en->fr", "fr->mixed", "en->mixed"} <= pairs
    categories = {q.category for q in dataset.answerable}
    expected = {"paraphrase", "cross_lingual", "fine_grained", "numeric", "multi_passage"}
    assert expected | {"long_page"} <= categories
    assert all(not q.category.startswith("unanswerable") for q in dataset.answerable)
    assert all(q.category.startswith("unanswerable") for q in dataset.unanswerable)
