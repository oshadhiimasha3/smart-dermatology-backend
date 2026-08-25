"""
Phase 4 — Recommendation Engine (rule-based / semantic-mapping style, as
described in Section 2.4.1 of the contextual report).

Not present in any of the three notebooks (they intentionally stop at the
fused diagnosis) — this module is the missing piece that turns
`final_condition` + `additional_concerns` into the personalized skincare
guidance promised in the project's aim & objectives.
"""
import json
import logging

from app import config

logger = logging.getLogger("smart_dermatology.recommend")

_db = None


def load():
    global _db
    with open(config.RECOMMENDATIONS_PATH) as f:
        _db = json.load(f)
    logger.info("Recommendation DB loaded with %d conditions", len(_db["conditions"]))
    return _db


def is_loaded() -> bool:
    return _db is not None


def build_recommendation(final_condition: str, additional_concerns: list[str], skin_type_hint: str | None = None):
    """
    Args:
        final_condition: the fused model's top predicted class (one of the 9 shared classes).
        additional_concerns: multi-label concerns detected by the text model
                              (e.g. ["itching", "flaking"]) — vocabulary is
                              whatever the training dataset defined.
        skin_type_hint: optional, if the user separately stated their skin type.

    Returns:
        dict ready to serialize straight into the API response.
    """
    entry = _db["conditions"].get(final_condition)

    if entry is None:
        # Defensive fallback — should not happen if fusion_meta classes match this DB
        return {
            "condition": final_condition,
            "possible_causes": [],
            "key_ingredients": [],
            "routine": [],
            "avoid": [],
            "see_a_doctor_if": "Consult a dermatologist for a condition not covered by this guide.",
            "product_types": [],
            "additional_concern_notes": {},
            "skin_type_context": skin_type_hint,
            "skin_type_note": None,
            "disclaimer": _db["disclaimer"],
        }

    concern_notes = {
        concern: _db.get("concern_addon_notes", {}).get(concern, _db["concern_addon_notes"]["default"])
        for concern in additional_concerns
    }

    skin_type_note = None
    if skin_type_hint:
        skin_type_note = _db.get("skin_type_notes", {}).get(skin_type_hint.strip().lower())

    return {
        "condition": final_condition,
        "possible_causes": entry["possible_causes"],
        "key_ingredients": entry["key_ingredients"],
        "routine": entry["routine"],
        "avoid": entry["avoid"],
        "see_a_doctor_if": entry["see_a_doctor_if"],
        "product_types": entry.get("product_types", []),
        "additional_concern_notes": concern_notes,
        "skin_type_context": skin_type_hint,
        "skin_type_note": skin_type_note,
        "disclaimer": _db["disclaimer"],
    }
