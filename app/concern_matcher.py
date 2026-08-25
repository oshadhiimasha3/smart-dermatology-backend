"""
Lay-language concern matcher — a rule-based safety net that runs alongside
the trained text model's concern head.

Why this exists: the concern head was trained on synthetic data where each
label's examples lean heavily on the literal label word (e.g. "blackheads").
A user typing "black dots on my nose" describes the same real-world concern
but shares no vocabulary with what the model saw in training, so it can
score near-zero for that label even though the concern is genuinely present.

This module does NOT replace the model. It runs independently on the same
raw description text and returns any concerns it recognises from a curated
lay-language vocabulary. main.py takes the UNION of both — the model can
still detect anything it's learned, and this catches common phrasings it
hasn't. Nothing the model already detects is removed or overridden.

Matching is case-insensitive, whole-phrase, word-boundary substring
matching (not fuzzy/semantic) — deliberately simple and auditable, since a
false positive here just adds an extra concern note, while a silent miss
is the exact failure mode this module exists to catch.
"""
import json
import logging
import re

from app import config

logger = logging.getLogger("smart_dermatology.concern_matcher")

_vocab = None
_compiled = None  # dict[concern_label, list[re.Pattern]]


def load():
    global _vocab, _compiled
    with open(config.CONCERN_VOCAB_PATH) as f:
        data = json.load(f)
    _vocab = data["vocabulary"]

    _compiled = {}
    for concern, phrases in _vocab.items():
        patterns = []
        for phrase in phrases:
            # Word-boundary match on the exact phrase, case-insensitive.
            # re.escape handles punctuation/apostrophes in phrases like "crow's feet".
            pattern = re.compile(r"\b" + re.escape(phrase.lower()) + r"\b")
            patterns.append(pattern)
        _compiled[concern] = patterns

    logger.info(
        "Concern vocabulary loaded: %d labels, %d total phrases",
        len(_compiled),
        sum(len(v) for v in _compiled.values()),
    )
    return _vocab


def is_loaded() -> bool:
    return _compiled is not None


def match_concerns(text: str) -> list[str]:
    """
    Args:
        text: the same raw free-text description passed to the text model.

    Returns:
        Sorted list of concern labels whose vocabulary matched somewhere
        in the text. Empty list if nothing matched or module isn't loaded.
    """
    if not text or _compiled is None:
        return []

    normalized = " ".join(text.lower().split())  # collapse whitespace
    matched = [
        concern
        for concern, patterns in _compiled.items()
        if any(p.search(normalized) for p in patterns)
    ]
    return sorted(matched)
