"""
Smart Dermatology Assistant — Backend API

Wires together the three trained components from the CIS013-3 project:
    1. Image model    (EfficientNetB3, Keras)      -> app/ml/image_model.py
    2. Text model     (DistilBERT multi-task, PT)  -> app/ml/text_model.py
    3. Fusion layer   (Weighted Avg / LR / MLP)     -> app/ml/fusion_model.py
plus a rule-based recommendation engine (Phase 4)   -> app/recommend.py

Run locally:
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

See README.md for exactly which files go in artifacts/, and for free
deployment instructions (Hugging Face Spaces).
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app import auth_routes, concern_matcher, config, database, history, recommend
from app.deps import get_current_user
from app.ml import fusion_model, image_model, text_model
from app.schemas import DiagnosisResponse, HealthResponse, SaveHistoryRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("smart_dermatology.main")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: load all three models + recommendation DB ONCE ──────────────
    t0 = time.time()
    logger.info("Loading models — this can take 15-60s on first boot...")
    try:
        image_model.load()
        text_model.load()
        fusion_model.load()
        recommend.load()
        concern_matcher.load()
    except Exception:
        logger.exception(
            "FAILED to load one or more models. Check that every file listed in "
            "README.md exists under artifacts/. The API will start but /diagnose "
            "will return 503 until this is fixed."
        )
    else:
        logger.info("All models loaded in %.1fs. Ready to serve.", time.time() - t0)

    try:
        await database.connect()
    except Exception:
        logger.exception(
            "FAILED to connect to MongoDB. Check MONGODB_URI in your .env file. "
            "The API will start but diagnosis history will not be saved until this is fixed."
        )

    yield
    await database.disconnect()
    logger.info("Shutting down.")


app = FastAPI(
    title="Smart Dermatology Assistant API",
    description="Multimodal (image + text) skin condition detection with personalized skincare recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", tags=["meta"])
def root():
    return FileResponse(os.path.join(_STATIC_DIR, "test_client.html"))


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    ready = image_model.is_loaded() and text_model.is_loaded() and fusion_model.is_loaded()
    return HealthResponse(
        status="ok" if ready else "models_not_loaded",
        image_model_loaded=image_model.is_loaded(),
        text_model_loaded=text_model.is_loaded(),
        fusion_loaded=fusion_model.is_loaded(),
        recommendations_loaded=recommend.is_loaded(),
        concern_vocab_loaded=concern_matcher.is_loaded(),
        fusion_method=fusion_model.get_method() if fusion_model.is_loaded() else None,
        classes=fusion_model.get_classes() if fusion_model.is_loaded() else None,
        database_connected=database.is_connected(),
    )


@app.post("/diagnose", response_model=DiagnosisResponse, tags=["diagnosis"])
async def diagnose(
    image: UploadFile = File(..., description="A clear photo of the affected skin area (JPG/PNG/WEBP)."),
    description: str = Form(..., description="Free-text description: symptoms, duration, skin type, etc."),
    skin_type: str | None = Form(None, description="Optional explicit skin type (oily/dry/combination/normal)."),
    current_user: dict = Depends(get_current_user),
):
    """
    Full multimodal pipeline: image -> EfficientNetB3, text -> DistilBERT,
    then fusion -> final condition, then rule-based recommendation lookup.

    Mirrors `diagnose()` from completed_-_Phase3_Fusion_Layer.ipynb, extended
    with the Phase 4 recommendation engine.
    """
    if not (image_model.is_loaded() and text_model.is_loaded() and fusion_model.is_loaded()):
        raise HTTPException(status_code=503, detail="Models are not loaded. Check server logs / artifacts folder.")

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{image.content_type}'. Use JPG, PNG, or WEBP.",
        )

    image_bytes = await image.read()
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image too large. Max {config.MAX_UPLOAD_MB}MB.")

    if not description or not description.strip():
        raise HTTPException(status_code=422, detail="`description` text field is required and cannot be empty.")

    try:
        img_probs, img_pred, img_conf = image_model.predict(image_bytes)
    except Exception as e:
        logger.exception("Image model inference failed")
        raise HTTPException(status_code=422, detail=f"Could not process the uploaded image: {e}")

    try:
        txt_probs, txt_pred, txt_conf, model_concerns = text_model.predict(description)
    except Exception as e:
        logger.exception("Text model inference failed")
        raise HTTPException(status_code=422, detail=f"Could not process the description text: {e}")

    # Keyword-matched concerns run independently of the model and only ADD to
    # what it found — nothing the model detected is ever dropped. See
    # app/concern_matcher.py for why this exists.
    keyword_concerns = concern_matcher.match_concerns(description) if concern_matcher.is_loaded() else []
    model_set, keyword_set = set(model_concerns), set(keyword_concerns)
    concerns = sorted(model_set | keyword_set)
    concern_sources = {
        c: ("model+keyword" if c in model_set and c in keyword_set else "model" if c in model_set else "keyword")
        for c in concerns
    }

    fused_probs = fusion_model.fuse(img_probs, txt_probs)
    classes = fusion_model.get_classes()
    fused_idx = int(fused_probs.argmax())
    final_condition = classes[fused_idx]
    final_confidence = round(float(fused_probs.max()) * 100, 2)

    recommendation = recommend.build_recommendation(final_condition, concerns, skin_type)

    response = DiagnosisResponse(
        final_condition=final_condition,
        final_confidence=final_confidence,
        additional_concerns=concerns,
        concern_sources=concern_sources,
        fusion_method=fusion_model.get_method(),
        image_model={"predicted_label": img_pred, "confidence": img_conf},
        text_model={"predicted_label": txt_pred, "confidence": txt_conf},
        probabilities={cls: round(float(p), 4) for cls, p in zip(classes, fused_probs)},
        recommendation=recommendation,
        disclaimer=(
            "This tool provides non-diagnostic, AI-assisted screening for general skincare "
            "guidance only. It is not a substitute for professional medical advice."
        ),
    )

    return response


@app.post("/history", tags=["history"], status_code=201)
async def create_history_entry(payload: SaveHistoryRequest, current_user: dict = Depends(get_current_user)):
    """Explicitly persist a diagnosis result the user chose to save."""
    if not database.is_connected():
        raise HTTPException(status_code=503, detail="Database is not connected.")
    saved = await history.save_diagnosis(
        final_condition=payload.response.final_condition,
        final_confidence=payload.response.final_confidence,
        description=payload.description,
        skin_type=payload.skin_type,
        response_payload=payload.response.model_dump(),
        user_id=str(current_user["_id"]),
    )
    return saved


@app.get("/history", tags=["history"])
async def get_history(limit: int = 20, skip: int = 0, current_user: dict = Depends(get_current_user)):
    """Most recent diagnoses first, scoped to the authenticated user."""
    if not database.is_connected():
        raise HTTPException(status_code=503, detail="Database is not connected.")
    limit = max(1, min(limit, 100))
    return await history.list_diagnoses(user_id=str(current_user["_id"]), limit=limit, skip=skip)


@app.get("/history/{diagnosis_id}", tags=["history"])
async def get_history_item(diagnosis_id: str, current_user: dict = Depends(get_current_user)):
    if not database.is_connected():
        raise HTTPException(status_code=503, detail="Database is not connected.")
    doc = await history.get_diagnosis(diagnosis_id)
    if doc is None or doc.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=404, detail="Diagnosis not found.")
    return doc


@app.delete("/history/{diagnosis_id}", tags=["history"])
async def delete_history_item(diagnosis_id: str, current_user: dict = Depends(get_current_user)):
    if not database.is_connected():
        raise HTTPException(status_code=503, detail="Database is not connected.")
    doc = await history.get_diagnosis(diagnosis_id)
    if doc is None or doc.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=404, detail="Diagnosis not found.")
    deleted = await history.delete_diagnosis(diagnosis_id)
    return {"deleted": deleted, "id": diagnosis_id}
