"""
CRUD helpers for the `diagnoses` collection.

Kept separate from main.py / database.py so the Mongo document shape lives
in exactly one place.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from app.database import get_db


def _serialize(doc: dict) -> dict:
    """Convert Mongo's ObjectId/datetime into JSON-friendly values."""
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    if isinstance(doc.get("created_at"), datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    return doc


async def save_diagnosis(
    *,
    final_condition: str,
    final_confidence: float,
    description: str,
    skin_type: Optional[str],
    response_payload: dict[str, Any],
    user_id: Optional[str] = None,
) -> dict:
    """Insert one diagnosis record and return it (serialized)."""
    db = get_db()
    doc = {
        "user_id": user_id,
        "final_condition": final_condition,
        "final_confidence": final_confidence,
        "description": description,
        "skin_type": skin_type,
        "response": response_payload,
        "created_at": datetime.now(timezone.utc),
    }
    result = await db.diagnoses.insert_one(doc)
    doc["_id"] = result.inserted_id
    serialized = _serialize(doc)

    # Push the serialized diagnosis under the user's document as well
    if user_id and ObjectId.is_valid(user_id):
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$push": {"diagnoses": serialized}}
        )

    return serialized


async def list_diagnoses(
    *, user_id: Optional[str] = None, limit: int = 20, skip: int = 0
) -> list[dict]:
    """Most recent diagnoses first, optionally filtered by user_id."""
    db = get_db()
    query = {"user_id": user_id} if user_id else {}
    cursor = db.diagnoses.find(query).sort("created_at", -1).skip(skip).limit(limit)
    return [_serialize(doc) async for doc in cursor]


async def get_diagnosis(diagnosis_id: str) -> Optional[dict]:
    db = get_db()
    if not ObjectId.is_valid(diagnosis_id):
        return None
    doc = await db.diagnoses.find_one({"_id": ObjectId(diagnosis_id)})
    return _serialize(doc) if doc else None


async def delete_diagnosis(diagnosis_id: str) -> bool:
    db = get_db()
    if not ObjectId.is_valid(diagnosis_id):
        return False

    # Pull the diagnosis from the user's document if it exists
    doc = await db.diagnoses.find_one({"_id": ObjectId(diagnosis_id)})
    if doc:
        user_id = doc.get("user_id")
        if user_id and ObjectId.is_valid(user_id):
            await db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$pull": {"diagnoses": {"id": diagnosis_id}}}
            )

    result = await db.diagnoses.delete_one({"_id": ObjectId(diagnosis_id)})
    return result.deleted_count == 1
