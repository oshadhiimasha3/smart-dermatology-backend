"""
MongoDB connection layer for the Smart Dermatology Assistant backend.

Uses Motor (the official async MongoDB driver) so it plays nicely with
FastAPI's async request handlers. The client is created once at app startup
(see lifespan in main.py) and reused for every request — never create a new
client per-request.

All Mongo settings come from environment variables (see .env.example).
"""
import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app import config

logger = logging.getLogger("smart_dermatology.database")

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect():
    """Open the MongoDB connection. Call once, at app startup."""
    global _client, _db
    if _client is not None:
        return  # already connected

    logger.info("Connecting to MongoDB at %s ...", config.MONGODB_DB_NAME)
    _client = AsyncIOMotorClient(config.MONGODB_URI, serverSelectionTimeoutMS=8000)

    # Fail fast if the URI/credentials are wrong, instead of failing silently
    # on the first real query later.
    await _client.admin.command("ping")

    _db = _client[config.MONGODB_DB_NAME]

    # Helpful index: fetch a user's most recent diagnoses quickly.
    await _db.diagnoses.create_index([("created_at", -1)])
    await _db.diagnoses.create_index("user_id")

    # Unique email constraint for the auth system.
    await _db.users.create_index("email", unique=True)

    logger.info("MongoDB connected: db='%s'", config.MONGODB_DB_NAME)


async def disconnect():
    """Close the MongoDB connection. Call once, at app shutdown."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed.")


def get_db() -> AsyncIOMotorDatabase:
    """Return the active database handle. Raises if connect() hasn't run yet."""
    if _db is None:
        raise RuntimeError(
            "MongoDB is not connected. Did app startup's lifespan call database.connect()?"
        )
    return _db


def is_connected() -> bool:
    return _db is not None
