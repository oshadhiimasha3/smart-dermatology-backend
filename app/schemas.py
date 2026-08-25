from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field

SkinType = Literal["oily", "dry", "combination", "sensitive", "normal"]


class ModelBreakdown(BaseModel):
    predicted_label: str
    confidence: float


class RecommendationBlock(BaseModel):
    condition: str
    possible_causes: list[str]
    key_ingredients: list[str]
    routine: list[str]
    avoid: list[str]
    see_a_doctor_if: str
    product_types: list[str] = []
    additional_concern_notes: dict
    skin_type_context: Optional[str] = None
    skin_type_note: Optional[str] = None
    disclaimer: str


class DiagnosisResponse(BaseModel):
    final_condition: str
    final_confidence: float
    additional_concerns: list[str]
    concern_sources: dict = {}
    fusion_method: str
    image_model: ModelBreakdown
    text_model: ModelBreakdown
    probabilities: dict
    recommendation: RecommendationBlock
    disclaimer: str
    history_id: str | None = None


class SaveHistoryRequest(BaseModel):
    description: str
    skin_type: str | None = None
    response: DiagnosisResponse


class HealthResponse(BaseModel):
    status: str
    image_model_loaded: bool
    text_model_loaded: bool
    fusion_loaded: bool
    recommendations_loaded: bool
    concern_vocab_loaded: bool = False
    fusion_method: Optional[str] = None
    classes: Optional[list[str]] = None
    database_connected: bool = False


# ---- Auth models -----------------------------------------------------------

class UserRegister(BaseModel):
    full_name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    skin_type: SkinType | None = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    id: str
    full_name: str
    email: str
    skin_type: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
