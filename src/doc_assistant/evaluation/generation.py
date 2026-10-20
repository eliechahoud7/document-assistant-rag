"""Évaluation de la génération : enregistrement (seule étape qui appelle le LLM), puis
notation hors ligne, déterministe et rejouable à partir de l'enregistrement.

La correction est estimée par la présence de « faits requis » (mots-clés, avec variantes
FR/EN) dans la réponse : c'est un indicateur approximatif. Une fiche de relecture permet à un
humain de corriger les verdicts (paraphrases, erreurs factuelles, appui des citations).
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from doc_assistant.domain.errors import LLMConfigurationError, LLMError, LLMQuotaError
from doc_assistant.domain.models import Chunk, SearchResult
from doc_assistant.domain.ports import AnswerGenerator
from doc_assistant.evaluation.dataset import DatasetError, EvalDataset, EvalQuestion, Reference
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.qa_service import QAService

RECORD_VERSION = 1
# Fenêtre de texte autour d'une citation de référence dans laquelle un fait requis doit figurer.
SUPPORT_WINDOW = 300
EXCERPT_LENGTH = 300
PROXY_WARNING = (
    "La correction est estimée par la présence de faits requis (mots-clés avec variantes "
    "FR/EN) : c'est un indicateur approximatif, à confirmer par la relecture humaine "
    "(paraphrases non reconnues, erreurs factuelles contenant le bon mot, appui des citations)."
)
CORRECTNESS = ("correct", "partial", "incorrect")
# Questions en partie sans réponse (champ facultatif ``unsupported`` de l'annexe).
EXPECTED_BEHAVIORS = ("acknowledge", "omit_ok")
# Formulations d'absence reconnues par défaut (FR/EN, comparées après ``normalize``) : une
# heuristique par mots-clés, qui ne comprend ni la portée de la négation ni le contexte.
DEFAULT_ABSENCE_MARKERS = (
    "not mention*",
    "not specif*",
    "not stat*",
    "not say",
    "not said",
    "not provide*",
    "not give*",
    "not contain*",
    "not cover*",
    "not discuss*",
    "not indicate*",
    "not include*",
    "not recommend*",
    "no recommendation",
    "no information",
    "no mention",
    "not available",
    "not in the documents",
    "not in the passages",
    "does not address",
    "do not address",
    "ne precise*",
    "ne mentionne*",
    "ne dit",
    "ne disent",
    "ne donne*",
    "ne contien*",
    "ne fourni*",
    "ne recommande*",
    "ne traite*",
    "ne parle*",
    "n'indique*",
    "n'aborde*",
    "n'evoque*",
    "pas mentionn*",
    "pas precis*",
    "pas indiqu*",
    "non mentionn*",
    "non precis*",
    "aucune information",
    "aucune indication",
    "aucune recommandation",
    "aucune mention",
    "ne figure pas",
    "ne figurent pas",
    # « gives no year », « recommend no value » : négation portée par « no ».
    "give no",
    "gives no",
    "provide no",
    "provides no",
    "recommend no",
    "recommends no",
    "mention no",
    "mentions no",
    "specify no",
    "specifies no",
    # « ne le précisent pas », « n'en parlent pas » : verbe séparé de « ne » (le « * » n'est
    # reconnu qu'en fin de variante, d'où les formes conjuguées explicites).
    "precise pas",
    "precisent pas",
    "mentionne pas",
    "mentionnent pas",
    "indique pas",
    "indiquent pas",
    "donne pas",
    "donnent pas",
    "dit pas",
    "disent pas",
    "recommande pas",
    "recommandent pas",
    "parle pas",
    "parlent pas",
    "traite pas",
    "traitent pas",
    "aborde pas",
    "abordent pas",
    "contient pas",
    "contiennent pas",
    "fournit pas",
    "fournissent pas",
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CITATION_MARKER_RE = re.compile(r"\[\d+\]")


# ---------------------------------------------------------------- normalisation et faits


def normalize(text: str) -> str:
    """Minuscules, sans accents (NFKD : « ² » devient « 2 »), espaces réduits."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


@lru_cache(maxsize=1024)
def _variant_pattern(alternative: str) -> re.Pattern[str]:
    """Mot ou expression entière ; un ``*`` final autorise un mot plus long (préfixe).

    Exemples : « zero » ne correspond pas à « nonzero » ; « 60 » ni à « 600 » ni à « 1.60 » ;
    « about 63 » pas à « about 63.25 » ; « polynom* » correspond à « polynomial ».
    """
    prefix = alternative.endswith("*")
    core = normalize(alternative[:-1] if prefix else alternative)
    if not core:
        raise ValueError(f"variante vide : {alternative!r}")
    left = ""
    if core[0].isalnum():
        left = r"(?<![0-9a-z])"
        if core[0].isdigit():  # un nombre ne commence pas après « 1. » ou « 1, »
            left += r"(?<![0-9][.,])"
    right = ""
    if not prefix and core[-1].isalnum():
        right = r"(?![0-9a-z])"
        if core[-1].isdigit():  # un nombre ne s'arrête pas avant « .5 » ou « ,5 »
            right += r"(?![.,][0-9])"
    return re.compile(left + re.escape(core) + right)


def fact_found(alternatives: Sequence[str], normalized_text: str) -> bool:
    return any(_variant_pattern(alt).search(normalized_text) for alt in alternatives)


# ---------------------------------------------------------------- annexe de référence


@dataclass(frozen=True, slots=True)
class UnsupportedPart:
    """Partie d'une question que les documents ne couvrent pas (question en partie répondable).

    ``required_facts`` de la réponse attendue ne portent que sur la partie répondable.
    ``expected_behavior`` : ``acknowledge`` (la réponse doit signaler l'absence) ou ``omit_ok``
    (taire la partie absente est accepté ; seule l'invention est pénalisée).
    ``forbidden_facts`` : réponses inventées plausibles, absentes des documents (vérifié au
    chargement). ``check_numbers`` : signale tout nombre de la réponse absent des passages
    retrouvés et de la question.
    """

    part: str
    expected_behavior: str
    absence_markers: tuple[str, ...]
    forbidden_facts: tuple[tuple[str, ...], ...]
    check_numbers: bool


@dataclass(frozen=True, slots=True)
class GoldAnswer:
    question_id: str
    expected_answer: str
    required_facts: tuple[tuple[str, ...], ...]
    support: str
    flags: tuple[str, ...]
    unsupported: UnsupportedPart | None = None


def _reference_windows(dataset: EvalDataset, question: EvalQuestion) -> list[str]:
    pages = {d.filename: d for d in dataset.documents}
    windows = []
    for ref in question.references:
        text = pages[ref.filename].pages[ref.page - 1].text
        windows.append(
            normalize(text[max(0, ref.start - SUPPORT_WINDOW) : ref.end + SUPPORT_WINDOW])
        )
    return windows


def load_gold(path: Path, dataset: EvalDataset) -> dict[str, GoldAnswer]:
    """Charge l'annexe et vérifie que chaque fait requis figure près d'une citation de référence."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path.name} : JSON invalide ({exc})") from exc
    if raw.get("dataset_sha256") != dataset.sha256:
        raise DatasetError(
            f"{path.name} : annexe écrite pour une autre version du jeu {dataset.name!r} "
            "(dataset_sha256 différent)"
        )
    answers = raw.get("answers")
    if not isinstance(answers, dict):
        raise DatasetError(f"{path.name} : champ 'answers' manquant")
    expected = {q.id for q in dataset.answerable}
    if set(answers) != expected:
        missing, extra = sorted(expected - set(answers)), sorted(set(answers) - expected)
        raise DatasetError(f"{path.name} : questions manquantes {missing}, en trop {extra}")

    gold: dict[str, GoldAnswer] = {}
    by_id = {q.id: q for q in dataset.answerable}
    for qid, item in answers.items():
        text, facts = item.get("expected_answer"), item.get("required_facts")
        support, flags = item.get("support"), item.get("flags", [])
        if not isinstance(text, str) or not text.strip():
            raise DatasetError(f"{qid} : 'expected_answer' vide")
        if (
            not isinstance(facts, list)
            or not facts
            or not all(
                isinstance(f, list)
                and f
                and all(isinstance(a, str) and a.strip().rstrip("*").strip() for a in f)
                for f in facts
            )
        ):
            raise DatasetError(f"{qid} : 'required_facts' doit être une liste de listes non vides")
        if support not in ("strong", "weak"):
            raise DatasetError(f"{qid} : 'support' doit valoir 'strong' ou 'weak'")
        if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
            raise DatasetError(f"{qid} : 'flags' doit être une liste de chaînes")
        if support == "weak" and not flags:
            raise DatasetError(
                f"{qid} : un élément 'weak' doit expliquer sa faiblesse dans 'flags'"
            )
        windows = _reference_windows(dataset, by_id[qid])
        for fact in facts:
            if not any(fact_found(fact, w) for w in windows):
                raise DatasetError(
                    f"{qid} : le fait requis {fact} ne figure dans aucun passage de référence"
                )
        unsupported = None
        if "unsupported" in item:
            unsupported = _load_unsupported(qid, item["unsupported"], dataset)
        gold[qid] = GoldAnswer(
            qid, text, tuple(tuple(f) for f in facts), support, tuple(flags), unsupported
        )
    return gold


def _is_variant_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(a, str) and a.strip().rstrip("*").strip() for a in value)
    )


def _load_unsupported(qid: str, raw: Any, dataset: EvalDataset) -> UnsupportedPart:
    """Valide le champ facultatif ``unsupported`` d'une réponse attendue."""
    if not isinstance(raw, dict):
        raise DatasetError(f"{qid} : 'unsupported' doit être un objet")
    part, behavior = raw.get("part"), raw.get("expected_behavior", "acknowledge")
    markers = raw.get("absence_markers", list(DEFAULT_ABSENCE_MARKERS))
    forbidden, check_numbers = raw.get("forbidden_facts", []), raw.get("check_numbers", True)
    if not isinstance(part, str) or not part.strip():
        raise DatasetError(f"{qid} : 'unsupported.part' doit décrire la partie sans réponse")
    if behavior not in EXPECTED_BEHAVIORS:
        raise DatasetError(f"{qid} : 'expected_behavior' doit être parmi {EXPECTED_BEHAVIORS}")
    if not _is_variant_list(markers):
        raise DatasetError(f"{qid} : 'absence_markers' doit être une liste non vide de chaînes")
    if not isinstance(forbidden, list) or not all(_is_variant_list(f) for f in forbidden):
        raise DatasetError(f"{qid} : 'forbidden_facts' doit être une liste de listes non vides")
    if not isinstance(check_numbers, bool):
        raise DatasetError(f"{qid} : 'check_numbers' doit être un booléen")
    # Un fait « interdit » qui figure dans les documents ne prouverait pas une invention.
    pages = [normalize(page.text) for document in dataset.documents for page in document.pages]
    for fact in forbidden:
        if any(fact_found(fact, page) for page in pages):
            raise DatasetError(
                f"{qid} : le fait interdit {fact} figure dans les documents ; il ne peut pas "
                "servir à détecter une invention"
            )
    return UnsupportedPart(
        part.strip(),
        behavior,
        tuple(markers),
        tuple(tuple(f) for f in forbidden),
        check_numbers,
    )


def load_overrides(path: Path, dataset: EvalDataset) -> dict[str, dict[str, Any]]:
    """Verdicts humains : {id: {"correctness": ..., "citation_support": bool, "note": ...}}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = {q.id for q in dataset.questions}
    if not isinstance(raw, dict) or not set(raw) <= ids:
        raise DatasetError(f"{path.name} : identifiants inconnus {sorted(set(raw) - ids)}")
    for qid, item in raw.items():
        if "correctness" in item and item["correctness"] not in CORRECTNESS:
            raise DatasetError(f"{qid} : 'correctness' doit être parmi {CORRECTNESS}")
        for key in ("citation_support", "absence_acknowledged", "invented"):
            if key in item and not isinstance(item[key], bool):
                raise DatasetError(f"{qid} : '{key}' doit être un booléen")
    return dict(raw)


# ---------------------------------------------------------------- enregistrement


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def latest_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Dernier enregistrement par question (une reprise peut réessayer une question)."""
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[record["question_id"]] = dict(record)
    return latest


def _passage_to_dict(rank: int, result: SearchResult) -> dict[str, Any]:
    c = result.chunk
    return {
        "rank": rank,
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "filename": c.filename,
        "page": c.page_number,
        "chunk_index": c.chunk_index,
        "start_char": c.start_char,
        "score": round(result.score, 6),
        "text": c.text,
    }


def _passage_to_chunk(item: Mapping[str, Any]) -> Chunk:
    return Chunk(
        item["chunk_id"],
        item["doc_id"],
        item["filename"],
        item["page"],
        item["chunk_index"],
        item["start_char"],
        item["text"],
    )


@dataclass
class RecordingSummary:
    recorded: int = 0
    skipped: int = 0
    errors: int = 0
    api_calls: int = 0
    stopped_on_quota: bool = False
    remaining: list[str] = field(default_factory=list)


def record_answers(
    dataset: EvalDataset,
    retriever: Retriever,
    generator: AnswerGenerator,
    output: Path,
    *,
    run_config: Mapping[str, Any],
    top_k: int = 5,
    context_neighbors: int = 0,
    delay_s: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] = lambda _line: None,
) -> RecordingSummary:
    """Interroge le LLM question par question et ajoute une ligne JSON par question.

    Reprise : les questions déjà enregistrées avec succès sont ignorées ; les erreurs sont
    retentées. Sur quota épuisé (HTTP 429), arrêt propre sans nouvelle tentative.
    """
    config = dict(run_config)
    existing = read_records(output)
    for previous in existing:
        if previous.get("dataset_sha256") != dataset.sha256 or previous.get("config") != config:
            raise ValueError(
                f"{output.name} contient un enregistrement d'un autre jeu ou d'une autre "
                "configuration : choisissez un autre fichier de sortie"
            )
    done = {qid for qid, r in latest_records(existing).items() if r["status"] == "ok"}
    service = QAService(retriever, generator, context_neighbors=context_neighbors)
    summary = RecordingSummary()
    output.parent.mkdir(parents=True, exist_ok=True)

    questions = list(dataset.questions)
    for position, question in enumerate(questions):
        if question.id in done:
            summary.skipped += 1
            progress(f"{question.id} : déjà enregistrée, ignorée")
            continue
        passages = service.retrieve(question.question, top_k=top_k)
        if passages and summary.api_calls and delay_s > 0:
            sleep(delay_s)  # rythme compatible avec les limites par minute du niveau gratuit
        record: dict[str, Any] = {
            "version": RECORD_VERSION,
            "question_id": question.id,
            "dataset": dataset.name,
            "dataset_sha256": dataset.sha256,
            "config": config,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "question": question.question,
            "retrieved": [_passage_to_dict(i, r) for i, r in enumerate(passages, start=1)],
        }
        try:
            answer = service.answer(question.question, passages).answer
        except LLMQuotaError as exc:
            summary.stopped_on_quota = True
            summary.remaining = [q.id for q in questions[position:] if q.id not in done]
            progress(f"{question.id} : quota atteint, arrêt ({exc})")
            break
        except LLMConfigurationError:
            raise
        except LLMError as exc:
            summary.api_calls += bool(passages)
            summary.errors += 1
            record |= {
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            progress(f"{question.id} : erreur {type(exc).__name__} (sera retentée à la reprise)")
        else:
            summary.api_calls += bool(passages)
            summary.recorded += 1
            record |= {
                "status": "ok",
                "answer": {
                    "abstained": answer.abstained,
                    "abstain_reason": answer.abstain_reason,
                    "text": answer.text,
                    "model": answer.model,
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                    "citations": [
                        {
                            "number": c.number,
                            "chunk_id": c.chunk.chunk_id,
                            "filename": c.filename,
                            "page": c.page_number,
                        }
                        for c in answer.citations
                    ],
                },
            }
            verdict = "abstention" if answer.abstained else f"{len(answer.citations)} citation(s)"
            progress(f"{question.id} : enregistrée ({verdict})")
        with output.open("a", encoding="utf-8") as stream:  # sauvegarde après chaque question
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


# ---------------------------------------------------------------- notation hors ligne


def _ref_ranks(references: Sequence[Reference], chunks: Sequence[Chunk]) -> list[int | None]:
    return [
        next((i for i, c in enumerate(chunks, start=1) if ref.matches(c)), None)
        for ref in references
    ]


def _score_question(
    question: EvalQuestion,
    record: Mapping[str, Any] | None,
    gold: GoldAnswer | None,
    override: Mapping[str, Any],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": question.id,
        "lang": question.lang,
        "category": question.category,
        "answerable": question.answerable,
        "question": question.question,
    }
    if record is None or record.get("status") != "ok":
        status = "missing" if record is None else "error"
        return base | {"status": status, "outcome": status}

    chunks = [_passage_to_chunk(p) for p in record["retrieved"]]
    by_id = {c.chunk_id: c for c in chunks}
    answer = record["answer"]
    citations = answer["citations"]
    valid = [
        c
        for c in citations
        if 1 <= c["number"] <= len(chunks)
        and chunks[c["number"] - 1].chunk_id == c["chunk_id"]
        and (chunks[c["number"] - 1].filename, chunks[c["number"] - 1].page_number)
        == (c["filename"], c["page"])
    ]
    cited = [by_id[c["chunk_id"]] for c in valid]
    ref_ranks = _ref_ranks(question.references, chunks)
    cited_ref = [any(ref.matches(c) for c in cited) for ref in question.references]
    result = base | {
        "status": "ok",
        "abstained": answer["abstained"],
        "abstain_reason": answer["abstain_reason"],
        "answer": answer["text"],
        "retrieved": [(p["filename"], p["page"]) for p in record["retrieved"]],
        "reference_ranks": ref_ranks,
        "retrieval_hit": any(r is not None for r in ref_ranks),
        "citations": [(c.filename, c.page_number) for c in cited],
        "invalid_citations": len(citations) - len(valid),
        "cited_reference_passages": sum(
            any(r.matches(c) for r in question.references) for c in cited
        ),
        "references_cited": cited_ref,
    }
    if not question.answerable:
        return result | {
            "outcome": "correct_abstention" if answer["abstained"] else "unsupported_answer"
        }

    assert gold is not None
    if answer["abstained"]:
        auto, facts, supported = "abstained", [False] * len(gold.required_facts), []
    else:
        text = normalize(answer["text"])
        facts = [fact_found(f, text) for f in gold.required_facts]
        auto = "correct" if all(facts) else "partial" if any(facts) else "incorrect"
        cited_text = normalize(" ".join(c.text for c in cited))
        supported = [
            fact_found(f, cited_text)
            for f, ok in zip(gold.required_facts, facts, strict=True)
            if ok
        ]
    final = auto if answer["abstained"] else override.get("correctness", auto)
    human_support = override.get("citation_support")

    if not result["retrieval_hit"]:
        outcome = (
            "retrieval_failure_abstained" if answer["abstained"] else "retrieval_failure_answered"
        )
    elif answer["abstained"]:
        outcome = "unjustified_abstention"
    elif final != "correct":
        outcome = "generation_failure"
    elif not any(cited_ref) or human_support is False:
        outcome = "citation_failure"
    else:
        outcome = "success"
    extra: dict[str, Any] = {}
    if gold.unsupported is not None:
        partial = _score_unsupported(gold.unsupported, question, record, final, override)
        if partial["invented"]:
            outcome = "invented_unsupported_part"
        elif outcome == "success" and partial["behavior"] == "absence_not_acknowledged":
            outcome = "absence_not_acknowledged"
        extra["partial"] = partial
    return (
        result
        | {
            "facts_matched": facts,
            "facts_supported_by_citations": supported,
            "correctness_auto": auto,
            "correctness": final,
            "human": dict(override) or None,
            "gold_support": gold.support,
            "gold_flags": list(gold.flags),
            "outcome": outcome,
        }
        | extra
    )


def _canonical_number(raw: str) -> str:
    number = raw.replace(",", ".")
    return number.rstrip("0").rstrip(".") if "." in number else number


def _numbers(text: str) -> set[str]:
    return {_canonical_number(n) for n in _NUMBER_RE.findall(text)}


def _score_unsupported(
    unsupported: UnsupportedPart,
    question: EvalQuestion,
    record: Mapping[str, Any],
    correctness: str,
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Comportement face à la partie sans réponse (heuristiques par mots-clés, approximatives).

    Invention détectée = un fait interdit figure dans la réponse, ou (``check_numbers``) un
    nombre de la réponse ne figure ni dans les passages retrouvés ni dans la question. Une
    invention formulée autrement n'est pas détectée.
    """
    answer = record["answer"]
    abstained = bool(answer["abstained"])
    # Apostrophe typographique unifiée ; « doesn't » devient « does not » (marqueurs en « not »).
    raw = _CITATION_MARKER_RE.sub(" ", answer["text"]).replace(chr(0x2019), "'")
    text = normalize(raw).replace("n't", " not")
    acknowledged = fact_found(unsupported.absence_markers, text)
    forbidden = [list(f) for f in unsupported.forbidden_facts if fact_found(f, text)]
    numbers: list[str] = []
    if unsupported.check_numbers and not abstained:
        context = " ".join([question.question, *(p["text"] for p in record["retrieved"])])
        numbers = sorted(_numbers(text) - _numbers(normalize(context)))
    acknowledged = bool(override.get("absence_acknowledged", acknowledged))
    invented = not abstained and bool(override.get("invented", bool(forbidden or numbers)))
    flagged = acknowledged or unsupported.expected_behavior == "omit_ok"
    if abstained:
        behavior = "abstained_entirely"
    elif invented:
        behavior = "invented_unsupported_part"
    elif correctness == "correct":
        behavior = "complete" if flagged else "absence_not_acknowledged"
    else:
        behavior = "supported_part_incomplete" if flagged else "incomplete_and_unacknowledged"
    return {
        "part": unsupported.part,
        "expected_behavior": unsupported.expected_behavior,
        "absence_acknowledged": acknowledged,
        "forbidden_facts_found": forbidden,
        "unsupported_numbers": numbers,
        "invented": invented,
        "behavior": behavior,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def score_run(
    records: Sequence[Mapping[str, Any]],
    dataset: EvalDataset,
    gold: Mapping[str, GoldAnswer],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Note un enregistrement sans aucun appel réseau ; même entrée, même rapport."""
    for record in records:
        if record.get("dataset_sha256") != dataset.sha256:
            raise ValueError("l'enregistrement ne correspond pas à cette version du jeu")
    latest = latest_records(records)
    overrides = overrides or {}
    rows = [
        _score_question(q, latest.get(q.id), gold.get(q.id), overrides.get(q.id, {}))
        for q in dataset.questions
    ]
    answerable = [r for r in rows if r["answerable"] and r["status"] == "ok"]
    unanswerable = [r for r in rows if not r["answerable"] and r["status"] == "ok"]
    answered = [r for r in answerable if not r["abstained"]]
    abstained = [r for r in answerable if r["abstained"]]

    def correctness_counts(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
        counts = Counter(r[key] for r in items)
        return {k: counts.get(k, 0) for k in (*CORRECTNESS, "abstained")}

    cited_total = sum(len(r["citations"]) for r in answered)
    cited_gold = sum(r["cited_reference_passages"] for r in answered)
    retrieved_refs = sum(rank is not None for r in answered for rank in r["reference_ranks"])
    retrieved_and_cited = sum(
        rank is not None and cited
        for r in answered
        for rank, cited in zip(r["reference_ranks"], r["references_cited"], strict=True)
    )
    matched = sum(sum(r["facts_matched"]) for r in answered)
    supported = sum(sum(r["facts_supported_by_citations"]) for r in answered)
    configs = {json.dumps(r.get("config"), sort_keys=True) for r in records}
    partial_rows = [r for r in answerable if "partial" in r]
    # Bloc présent seulement si l'annexe utilise 'unsupported' : rapports antérieurs inchangés.
    partial_block = (
        {
            "partial_questions": {
                "n": len(partial_rows),
                "behaviors": dict(Counter(r["partial"]["behavior"] for r in partial_rows)),
                "absence_acknowledged": sum(
                    r["partial"]["absence_acknowledged"] for r in partial_rows
                ),
                "invented": [r["id"] for r in partial_rows if r["partial"]["invented"]],
            }
        }
        if partial_rows
        else {}
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "warning": PROXY_WARNING,
        "dataset": {"name": dataset.name, "sha256": dataset.sha256},
        "run_config": [json.loads(c) for c in sorted(configs)],
        "coverage": {
            "questions": len(rows),
            "scored": sum(r["status"] == "ok" for r in rows),
            "missing": [r["id"] for r in rows if r["status"] == "missing"],
            "errors": [r["id"] for r in rows if r["status"] == "error"],
        },
        "answerable": {
            "n": len(answerable),
            "answered": len(answered),
            "abstained": len(abstained),
            "abstained_justified": sum(not r["retrieval_hit"] for r in abstained),
            "abstained_unjustified": sum(r["retrieval_hit"] for r in abstained),
            "correctness_auto": correctness_counts(answerable, "correctness_auto"),
            "correctness": correctness_counts(answerable, "correctness"),
            "by_language": {
                lang: correctness_counts(
                    [r for r in answerable if r["lang"] == lang], "correctness"
                )
                for lang in sorted({r["lang"] for r in answerable})
            },
            "by_retrieval": {
                "reference_in_top_k": correctness_counts(
                    [r for r in answerable if r["retrieval_hit"]], "correctness"
                ),
                "reference_missed": correctness_counts(
                    [r for r in answerable if not r["retrieval_hit"]], "correctness"
                ),
            },
        },
        "citations": {
            "total": cited_total,
            "invalid": sum(r["invalid_citations"] for r in rows if r["status"] == "ok"),
            "reference_precision": _rate(cited_gold, cited_total),
            "reference_recall": _rate(retrieved_and_cited, retrieved_refs),
            "fact_support": _rate(supported, matched),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "correct_abstentions": sum(r["abstained"] for r in unanswerable),
            "unsupported_answers": sum(not r["abstained"] for r in unanswerable),
        },
        "multi_passage": {
            r["id"]: {
                "reference_ranks": r["reference_ranks"],
                "references_cited": r["references_cited"],
                "facts_matched": r["facts_matched"],
                "correctness": r["correctness"],
            }
            for r in answerable
            if r["category"] == "multi_passage"
        },
        **partial_block,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "questions": rows,
    }


# ---------------------------------------------------------------- restitution


def format_generation_summary(report: Mapping[str, Any]) -> str:
    a, c, u = report["answerable"], report["citations"], report["unanswerable"]
    cov = report["coverage"]
    lines = [
        f"Jeu : {report['dataset']['name']} — {cov['scored']}/{cov['questions']} questions notées"
        + (f" ; manquantes {cov['missing']}" if cov["missing"] else "")
        + (f" ; en erreur {cov['errors']}" if cov["errors"] else ""),
        f"ATTENTION : {report['warning']}",
        "",
        f"Questions répondables (n={a['n']}) : {a['answered']} réponses, {a['abstained']} "
        f"abstentions ({a['abstained_justified']} justifiées, {a['abstained_unjustified']} non "
        "justifiées)",
        f"  correction (après relecture éventuelle) : {a['correctness']}",
        f"  correction automatique                  : {a['correctness_auto']}",
        f"  passage de référence retrouvé           : {a['by_retrieval']['reference_in_top_k']}",
        f"  passage de référence manqué             : {a['by_retrieval']['reference_missed']}",
        f"Citations : {c['total']} (invalides : {c['invalid']}) ; précision vs référence "
        f"{c['reference_precision']} ; rappel des références retrouvées {c['reference_recall']} ; "
        f"faits appuyés par les passages cités {c['fact_support']}",
        f"Questions sans réponse (n={u['n']}) : {u['correct_abstentions']} abstentions correctes, "
        f"{u['unsupported_answers']} réponses non fondées",
    ]
    if "partial_questions" in report:
        p = report["partial_questions"]
        lines.append(
            f"Questions en partie répondables (n={p['n']}) : {p['behaviors']} ; absence signalée "
            f"{p['absence_acknowledged']}/{p['n']} ; inventions repérées {p['invented']} "
            "(heuristique : faits interdits et nombres absents des passages)"
        )
    lines.append(f"Issues : {report['outcomes']}")
    return "\n".join(lines)


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:EXCERPT_LENGTH] + "…" if len(flat) > EXCERPT_LENGTH else flat


def format_review_sheet(
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    gold: Mapping[str, GoldAnswer],
) -> str:
    """Fiche Markdown pour la relecture humaine des verdicts automatiques."""
    latest = latest_records(records)
    lines = [
        f"# Relecture — {report['dataset']['name']}",
        "",
        f"> {report['warning']}",
        "",
        "Pour corriger un verdict, créez un fichier JSON passé à `score-answers --overrides` :",
        "",
        "```json",
        '{"q03": {"correctness": "correct", "citation_support": true, "note": "paraphrase"}}',
        "```",
        "",
    ]
    for row in report["questions"]:
        lines += [
            f"## {row['id']} — {row['category']} ({row['lang']})",
            "",
            f"**Question** : {row['question']}",
            "",
        ]
        if row["status"] != "ok":
            lines += [f"**Statut** : {row['status']}", ""]
            continue
        if row["answerable"]:
            g = gold[row["id"]]
            lines += [f"**Réponse attendue** : {g.expected_answer}"]
            if g.flags:
                lines += [f"**Limites de la référence ({g.support})** : " + " ; ".join(g.flags)]
            lines += [
                "**Faits requis** : "
                + " ; ".join(
                    f"{'✔' if ok else '✘'} {' | '.join(alts)}"
                    for alts, ok in zip(g.required_facts, row["facts_matched"], strict=True)
                ),
                "",
            ]
        verdict = "abstention" if row["abstained"] else "réponse"
        lines += [
            f"**Recherche** : passage de référence retrouvé = {row['retrieval_hit']} "
            f"(rangs {row['reference_ranks']})"
            if row["answerable"]
            else "**Recherche** : —",
            f"**Génération ({verdict})** : {row['answer'] or '(vide)'}",
        ]
        if row["abstained"]:
            lines += [f"**Raison** : {row['abstain_reason']}"]
        record = latest[row["id"]]
        chunks = {p["chunk_id"]: p for p in record["retrieved"]}
        for citation in record["answer"]["citations"]:
            passage = chunks.get(citation["chunk_id"])
            excerpt = _excerpt(passage["text"]) if passage else "(passage introuvable)"
            source = f"[{citation['number']}] {citation['filename']}, page {citation['page']}"
            lines += [f"- {source} : {excerpt}"]
        auto = row.get("correctness_auto", "—")
        lines += ["", f"**Verdict automatique** : {auto} → issue `{row['outcome']}`"]
        if "partial" in row:
            p = row["partial"]
            lines += [
                f"**Partie sans réponse** : {p['part']} (attendu : {p['expected_behavior']}) — "
                f"absence signalée = {p['absence_acknowledged']} ; faits interdits trouvés = "
                f"{p['forbidden_facts_found']} ; nombres absents des passages = "
                f"{p['unsupported_numbers']} → `{p['behavior']}` (corrigeable avec "
                '`"absence_acknowledged"` / `"invented"` dans les verdicts humains)',
            ]
        lines += [
            "",
            "**Relecture** : correction = ____ ; "
            "citations appuient la réponse = oui / non ; note = ____",
            "",
        ]
    return "\n".join(lines)
