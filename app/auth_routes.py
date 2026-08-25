"""
/auth/register, /auth/login, /auth/me
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app import security, users
from app.deps import get_current_user
from app.schemas import TokenResponse, UserLogin, UserPublic, UserRegister

logger = logging.getLogger("smart_dermatology.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: UserRegister):
    existing = await users.get_user_by_email(payload.email)
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")

    password_hash = security.hash_password(payload.password)
    try:
        user_doc = await users.create_user(
            full_name=payload.full_name.strip(),
            email=payload.email,
            password_hash=password_hash,
            skin_type=payload.skin_type,
        )
    except Exception as e:
        # Catches a race where two requests register the same email at once
        # (unique index on users.email rejects the second insert).
        logger.warning("Registration insert failed, likely duplicate email: %s", e)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")

    token = security.create_access_token(str(user_doc["_id"]))
    return TokenResponse(access_token=token, user=UserPublic(**users.to_public(user_doc)))


@router.post("/login", response_model=TokenResponse)
async def login(payload: UserLogin):
    user_doc = await users.get_user_by_email(payload.email)
    if user_doc is None or not security.verify_password(payload.password, user_doc["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")

    token = security.create_access_token(str(user_doc["_id"]))
    return TokenResponse(access_token=token, user=UserPublic(**users.to_public(user_doc)))


@router.get("/me", response_model=UserPublic)
async def me(current_user: dict = Depends(get_current_user)):
    return UserPublic(**users.to_public(current_user))
