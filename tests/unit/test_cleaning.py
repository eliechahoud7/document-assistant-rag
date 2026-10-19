import pytest

from doc_assistant.ingestion.cleaning import clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Une ef\ufb01cace con\ufb01guration", "Une efficace configuration"),  # ligatures
        ("infor-\nmation", "information"),  # césure
        ("Paris-\nSaclay", "Paris-\nSaclay"),  # majuscule après le tiret : on garde
        ("a\u00a0b\u202fc", "a b c"),  # espaces insécables
        ("a  \t  b", "a b"),
        ("ligne 1   \n   ligne 2", "ligne 1\nligne 2"),
        ("§1\n\n\n\n§2", "§1\n\n§2"),  # paragraphes conservés, lignes vides réduites
        ("a\r\nb\rc", "a\nb\nc"),
        ("nul\x00 et\x07 contrôle", "nul et contrôle"),
        ("  \n  texte  \n  ", "texte"),
        ("", ""),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_accents_are_preserved() -> None:
    text = "Été, œuvre, naïveté, garçon, À bientôt"

    assert clean_text(text) == text


def test_cleaning_is_idempotent() -> None:
    raw = "Une  ef\ufb01cace\u00a0infor-\nmation\n\n\n\nfin  "
    once = clean_text(raw)

    assert clean_text(once) == once
