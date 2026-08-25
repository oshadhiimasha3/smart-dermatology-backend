"""
CRUD helpers for the `users` collection.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from app.database import get_db


def _serialize_public(doc: dict) -> dict:
    """User fields safe to send to the client — NEVER include password_hash."""
    return {
        "id": str(doc["_id"]),
        "full_name": doc["full_name"],
        "email": doc["email"],
        "skin_type": doc.get("skin_type"),
    }


async def create_user(*, full_name: str, email: str, password_hash: str, skin_type: Optional[str]) -> dict:
    db = get_db()
    doc = {
        "full_name": full_name,
        "email": email.lower().strip(),
        "password_hash": password_hash,
        "skin_type": skin_type,
        "created_at": datetime.now(timezone.utc),
    }
    result = await db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def get_user_by_email(email: str) -> Optional[dict]:
    db = get_db()
    return await db.users.find_one({"email": email.lower().strip()})


async def get_user_by_id(user_id: str) -> Optional[dict]:
    db = get_db()
    if not ObjectId.is_valid(user_id):
        return None
    return await db.users.find_one({"_id": ObjectId(user_id)})


def to_public(doc: dict) -> dict:
    return _serialize_public(doc)
