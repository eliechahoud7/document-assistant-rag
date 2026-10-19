"""Génération de petits PDF de test à texte sélectionnable, avec pypdf uniquement."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import ContentStream, DictionaryObject, NameObject


def _escape_pdf_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: Sequence[Sequence[str]]) -> Path:
    """Écrit un PDF minimal : une entrée de ``pages`` par page, une chaîne par ligne.

    Une page sans ligne est une page blanche (aucun texte extractible). Le texte est
    encodé en WinAnsi (cp1252), ce qui couvre les accents du français.
    """
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    writer = PdfWriter()
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        if not lines:
            continue
        shown = " T* ".join(f"({_escape_pdf_string(line)}) Tj" for line in lines)
        content = ContentStream(None, writer)
        content.set_data(f"BT /F1 11 Tf 14 TL 72 720 Td {shown} ET".encode("cp1252"))
        page.replace_contents(content)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
    with path.open("wb") as stream:
        writer.write(stream)
    return path
