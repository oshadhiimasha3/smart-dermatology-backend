"""
ONNX Conversion Script for Smart Dermatology Backend
=====================================================

Run this ONCE on your LOCAL computer (where TensorFlow, PyTorch, and
transformers are already installed) to convert both trained models into
lightweight .onnx files.

Usage:
    cd smart_dermatology_backend
    python convert_to_onnx.py

After this script finishes you will have two new files:
    artifacts/image/image_model.onnx
    artifacts/text/text_model.onnx

These .onnx files replace the heavy .keras and .pt originals for deployment.
"""

import os
import json
import sys

# ---------------------------------------------------------------------------
# Paths (match config.py defaults)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

IMAGE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "image", "skin_model_efficientnetb3_finetuned.keras")
IMAGE_META_PATH = os.path.join(ARTIFACTS_DIR, "image", "image_model_meta.json")
IMAGE_ONNX_PATH = os.path.join(ARTIFACTS_DIR, "image", "image_model.onnx")

TEXT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "text", "distilbert_dermatology_text_model.pt")
TEXT_META_PATH = os.path.join(ARTIFACTS_DIR, "text", "text_model_meta.json")
TEXT_LABELS_PATH = os.path.join(ARTIFACTS_DIR, "text", "text_label_classes.json")
TEXT_ONNX_PATH = os.path.join(ARTIFACTS_DIR, "text", "text_model.onnx")


def convert_image_model():
    """Convert the Keras EfficientNetB3 model to ONNX."""
    print("\n" + "=" * 60)
    print("  STEP 1: Converting IMAGE model (EfficientNetB3 Keras to ONNX)")
    print("=" * 60)

    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        print("ERROR: You need tf2onnx and tensorflow installed.")
        print("Run:  pip install tf2onnx tensorflow-cpu")
        sys.exit(1)

    # Load metadata to get the expected image size
    with open(IMAGE_META_PATH) as f:
        meta = json.load(f)
    img_size = tuple(meta["img_size"])  # e.g. (300, 300)

    print(f"  Loading Keras model from: {IMAGE_MODEL_PATH}")
    print(f"  Expected input shape: (1, {img_size[0]}, {img_size[1]}, 3)")
    model = tf.keras.models.load_model(IMAGE_MODEL_PATH)

    # Define the input signature for tf2onnx
    # Shape: [batch_size, height, width, channels]
    input_spec = (tf.TensorSpec((1, img_size[0], img_size[1], 3), tf.float32, name="input"),)

    print(f"  Exporting to: {IMAGE_ONNX_PATH}")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec,
        opset=17,
        output_path=IMAGE_ONNX_PATH,
    )

    size_mb = os.path.getsize(IMAGE_ONNX_PATH) / (1024 * 1024)
    print(f"  [SUCCESS] Image model converted successfully! ({size_mb:.1f} MB)")


def convert_text_model():
    """Convert the custom PyTorch DistilBERT multi-task model to ONNX."""
    print("\n" + "=" * 60)
    print("  STEP 2: Converting TEXT model (DistilBERT PyTorch to ONNX)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: You need torch installed.")
        print("Run:  pip install torch")
        sys.exit(1)

    try:
        from transformers import AutoModel
    except ImportError:
        print("ERROR: You need transformers installed.")
        print("Run:  pip install transformers")
        sys.exit(1)

    # Load metadata
    with open(TEXT_META_PATH) as f:
        meta = json.load(f)

    with open(TEXT_LABELS_PATH) as f:
        label_data = json.load(f)

    max_length = int(meta.get("max_length", 128))
    base_arch = meta.get("base_architecture", "distilbert-base-uncased")
    dropout = float(meta.get("dropout", 0.3))
    num_main_classes = len(label_data["main_condition_classes"])
    num_concerns = len(label_data["additional_concern_classes"])

    print(f"  Base architecture: {base_arch}")
    print(f"  Max sequence length: {max_length}")
    print(f"  Main classes: {num_main_classes}, Concern classes: {num_concerns}")

    # -----------------------------------------------------------------------
    # Rebuild the EXACT same architecture from text_model.py
    # -----------------------------------------------------------------------
    class DermatologyTextModel(nn.Module):
        def __init__(self, model_name, num_main_classes, num_concerns, dropout=0.3):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.main_condition_head = nn.Linear(hidden_size, num_main_classes)
            self.concerns_head = nn.Linear(hidden_size, num_concerns)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.dropout(out.last_hidden_state[:, 0])
            main_logits = self.main_condition_head(pooled)
            concern_logits = self.concerns_head(pooled)
            return main_logits, concern_logits

    # Build the model
    print(f"  Building model architecture...")
    device = torch.device("cpu")
    model = DermatologyTextModel(
        model_name=base_arch,
        num_main_classes=num_main_classes,
        num_concerns=num_concerns,
        dropout=dropout,
    )

    # Load trained weights
    print(f"  Loading trained weights from: {TEXT_MODEL_PATH}")
    state_dict = torch.load(TEXT_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # -----------------------------------------------------------------------
    # Create dummy inputs matching the tokenizer output shape
    # -----------------------------------------------------------------------
    dummy_input_ids = torch.zeros(1, max_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)

    print(f"  Exporting to: {TEXT_ONNX_PATH}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        TEXT_ONNX_PATH,
        export_params=True,
        opset_version=17,
        input_names=["input_ids", "attention_mask"],
        output_names=["main_logits", "concern_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "main_logits": {0: "batch_size"},
            "concern_logits": {0: "batch_size"},
        },
    )

    size_mb = os.path.getsize(TEXT_ONNX_PATH) / (1024 * 1024)
    print(f"  [SUCCESS] Text model converted successfully! ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("Smart Dermatology — ONNX Model Conversion")
    print("==========================================")
    print(f"Artifacts directory: {ARTIFACTS_DIR}\n")

    # Verify source files exist
    missing = []
    for path, name in [
        (IMAGE_MODEL_PATH, "Image model (.keras)"),
        (IMAGE_META_PATH, "Image model metadata"),
        (TEXT_MODEL_PATH, "Text model (.pt)"),
        (TEXT_META_PATH, "Text model metadata"),
        (TEXT_LABELS_PATH, "Text label classes"),
    ]:
        if not os.path.exists(path):
            missing.append(f"  [ERROR] {name}: {path}")
    
    if missing:
        print("ERROR: The following required files are missing:")
        for m in missing:
            print(m)
        sys.exit(1)

    convert_image_model()
    convert_text_model()

    print("\n" + "=" * 60)
    print("  [SUCCESS] ALL CONVERSIONS COMPLETE!")
    print("=" * 60)
    print(f"\n  Image ONNX: {IMAGE_ONNX_PATH}")
    print(f"  Text  ONNX: {TEXT_ONNX_PATH}")
    print("\n  Next steps:")
    print("    1. Test locally with the updated ONNX-based backend code")
    print("    2. Push everything (including .onnx files) to GitHub")
    print("    3. Deploy on Koyeb!")
