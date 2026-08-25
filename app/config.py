"""
Central configuration for the Smart Dermatology Assistant backend.

Every path below can be overridden with an environment variable of the same
name (useful when deploying to Hugging Face Spaces / Docker, where you may
mount or bake the artifacts folder differently). Defaults assume the folder
layout described in README.md, rooted at ARTIFACTS_DIR.
"""
import os

from dotenv import load_dotenv

# Load variables from a .env file in the project root (if present) into the
# environment, BEFORE any os.environ.get() calls below run.
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(BASE_DIR, "artifacts"))

# ---- Image model (EfficientNetB3) -------------------------------------------
IMAGE_MODEL_PATH = os.environ.get(
    "IMAGE_MODEL_PATH",
    os.path.join(ARTIFACTS_DIR, "image", "skin_model_efficientnetb3_finetuned.keras"),
)
IMAGE_MODEL_ONNX_PATH = os.environ.get(
    "IMAGE_MODEL_ONNX_PATH",
    os.path.join(ARTIFACTS_DIR, "image", "image_model.onnx"),
)
IMAGE_LABELS_PATH = os.environ.get(
    "IMAGE_LABELS_PATH", os.path.join(ARTIFACTS_DIR, "image", "image_label_classes.json")
)
IMAGE_META_PATH = os.environ.get(
    "IMAGE_META_PATH", os.path.join(ARTIFACTS_DIR, "image", "image_model_meta.json")
)

# ---- Text model (DistilBERT multi-task) -------------------------------------
TEXT_MODEL_PATH = os.environ.get(
    "TEXT_MODEL_PATH",
    os.path.join(ARTIFACTS_DIR, "text", "distilbert_dermatology_text_model.pt"),
)
TEXT_MODEL_ONNX_PATH = os.environ.get(
    "TEXT_MODEL_ONNX_PATH",
    os.path.join(ARTIFACTS_DIR, "text", "text_model.onnx"),
)
TEXT_LABELS_PATH = os.environ.get(
    "TEXT_LABELS_PATH", os.path.join(ARTIFACTS_DIR, "text", "text_label_classes.json")
)
TEXT_META_PATH = os.environ.get(
    "TEXT_META_PATH", os.path.join(ARTIFACTS_DIR, "text", "text_model_meta.json")
)
TOKENIZER_DIR = os.environ.get(
    "TOKENIZER_DIR", os.path.join(ARTIFACTS_DIR, "text", "tokenizer")
)

# ---- Fusion layer -------------------------------------------------------------
FUSION_DIR = os.environ.get("FUSION_DIR", os.path.join(ARTIFACTS_DIR, "fusion"))
FUSION_META_PATH = os.path.join(FUSION_DIR, "fusion_meta.json")
FUSION_MLP_PATH = os.path.join(FUSION_DIR, "fusion_mlp.pt")
FUSION_LR_PATH = os.path.join(FUSION_DIR, "fusion_lr.pkl")
FUSION_SCALER_PATH = os.path.join(FUSION_DIR, "fusion_scaler.pkl")

# ---- Recommendations (Phase 4 — rule-based, shipped with this backend) ------
RECOMMENDATIONS_PATH = os.environ.get(
    "RECOMMENDATIONS_PATH", os.path.join(BASE_DIR, "app", "recommendations_db.json")
)

# ---- Concern vocabulary (lay-language keyword fallback for the concern head) --
CONCERN_VOCAB_PATH = os.environ.get(
    "CONCERN_VOCAB_PATH", os.path.join(BASE_DIR, "app", "concern_vocabulary.json")
)

# ---- API behaviour ------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# ---- MongoDB --------------------------------------------------------------
# MONGODB_URI examples:
#   Local:        mongodb://localhost:27017
#   Atlas (cloud): mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "smart_dermatology")

# ---- Auth / JWT --------------------------------------------------------
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-only-insecure-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "43200"))  # 30 days

if JWT_SECRET_KEY == "dev-only-insecure-secret-change-me":
    import logging
    logging.getLogger("smart_dermatology.config").warning(
        "JWT_SECRET_KEY is not set in .env — using an insecure default. "
        "Set a real random secret before deploying anywhere."
    )
