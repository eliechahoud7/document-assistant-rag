"""Nettoyage minimal du texte extrait des PDF."""

from __future__ import annotations

import re
import unicodedata

# Caractères de contrôle, sauf tabulation (\t) et saut de ligne (\n).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Espaces Unicode exotiques (insécable, fine, etc.) ramenés à une espace simple.
_UNICODE_SPACES_RE = re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
# Césure en fin de ligne suivie d'une minuscule : "infor-\nmation" -> "information".
_HYPHENATION_RE = re.compile(r"(\w)-\n(?=[a-zà-öø-ÿœ])")
_MULTI_SPACES_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise le texte d'une page sans en changer le sens.

    - NFKC : décompose les ligatures typographiques ("\ufb01" -> "fi") fréquentes dans les PDF ;
    - supprime les caractères de contrôle et uniformise espaces et fins de ligne ;
    - recolle les mots coupés par une césure en fin de ligne ;
    - conserve les paragraphes (au plus une ligne vide consécutive).

    Limite connue : un mot composé coupé juste après son tiret ("porte-\\nmonnaie")
    est recollé à tort ("portemonnaie").
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _UNICODE_SPACES_RE.sub(" ", text)
    text = _MULTI_SPACES_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _HYPHENATION_RE.sub(r"\1", text)
    text = _MULTI_NEWLINES_RE.sub("\n\n", text)
    return text.strip()
