# Smart Dermatology Assistant — Backend

Production-style FastAPI backend that wires together your three trained
components (image model → text model → fusion layer) into one `/diagnose`
endpoint, plus a rule-based recommendation engine (the "Phase 4" piece your
fusion notebook explicitly left for later).

This is the exact code path your `completed_-_Phase3_Fusion_Layer.ipynb`
`diagnose()` function runs — just moved out of Colab and into a real API
server, with a recommendation lookup added on top.

---

## 1. Get your trained artifacts out of Colab and into this project

You already have everything — it's sitting in your Google Drive
`smart_dermatology_models/` folder (per the fusion notebook's own
instructions) and/or in the zip files each notebook downloads. You need to
place them into `artifacts/` **exactly** like this:

```
artifacts/
├── image/
│   ├── skin_model_efficientnetb3_finetuned.keras   # from image notebook, Stage 11
│   ├── image_label_classes.json                     # = label_classes.json, RENAMED
│   └── image_model_meta.json
├── text/
│   ├── distilbert_dermatology_text_model.pt          # from text notebook, Stage 13
│   ├── text_label_classes.json                       # = label_classes.json, RENAMED
│   ├── text_model_meta.json
│   └── tokenizer/                                     # the whole folder, as-is
│       ├── vocab.txt
│       ├── tokenizer_config.json
│       └── ... (whatever files save_pretrained wrote)
└── fusion/
    ├── fusion_meta.json          # always present — tells the backend which method won
    ├── fusion_lr.pkl             # only needed if fusion_meta.json says "Stacked LR"
    ├── fusion_scaler.pkl         # needed for "Stacked LR" OR "Stacked MLP"
    └── fusion_mlp.pt             # only needed if fusion_meta.json says "Stacked MLP"
```

**Important renames** (both the image and text notebooks save a file called
plain `label_classes.json` — you must rename them so they don't collide):
- Image notebook's `label_classes.json` → `artifacts/image/image_label_classes.json`
- Text notebook's `label_classes.json` → `artifacts/text/text_label_classes.json`

Open `artifacts/fusion/fusion_meta.json` and check the `"best_method"` field —
that tells you which of the three fusion files above you actually need (you
don't need all of them, only the ones matching the winning method).

If you haven't run the fusion notebook fully to completion yet (Stage 8,
"Save Fusion Artefacts"), run it now in Colab end-to-end — it needs the two
`.npy` test-prediction files from each of the other two notebooks (the
notebooks already save these automatically at their final evaluation cell,
per the code you already have).

---

## 2. Run it locally

```bash
cd smart_dermatology_backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

First boot loads all three models (15–60s depending on your machine — the
DistilBERT and EfficientNetB3 weights aren't tiny). Watch the terminal logs;
you'll see `All models loaded in X.Xs. Ready to serve.`

Then:
- Open **http://localhost:8000/docs** — interactive Swagger UI, you can test
  `/diagnose` directly from the browser (upload a file + type text).
- Open **http://localhost:8000/health** — confirms all three components + the
  recommendation DB loaded correctly, and shows which fusion method is active.
- Open `static/test_client.html` directly in your browser (double-click it,
  no server needed) for a simple upload form wired to your running API —
  good for a quick visual "does the whole pipeline work" check before you
  build the real mobile/web frontend.

**If `/health` shows something not loaded:** check the terminal error log —
it will name the exact missing file path from the layout in Step 1.

---

## 3. Host it for free (so your app can call it without running anything locally)

**Recommended: Hugging Face Spaces (Docker SDK) — free CPU tier, no card needed.**

1. Create a free account at huggingface.co, then **New Space** →
   SDK: **Docker** → Hardware: **CPU basic (free)**.
2. Push this entire folder (including `artifacts/`) to the Space's git repo:
   ```bash
   git init
   git lfs install
   git lfs track "*.keras" "*.pt" "*.pkl"      # large binary files
   git add .gitattributes
   git add .
   git commit -m "Smart Dermatology Assistant backend"
   git remote add origin https://huggingface.co/spaces/<your-username>/<space-name>
   git push origin main
   ```
   (Use `git lfs` for the model weights — GitHub/HF reject large files
   otherwise. `.keras` for EfficientNetB3 will likely be 40–90MB.)
3. The Space auto-builds from your `Dockerfile` and serves on port 7860
   internally — nothing to configure, that's already set up in this project.
4. Your API is now live at `https://<your-username>-<space-name>.hf.space`.
   Test with:
   ```bash
   curl https://<your-username>-<space-name>.hf.space/health
   ```

**Free tier caveats to know about:**
- Free CPU Spaces sleep after inactivity and take ~20-30s to wake on the next
  request — fine for a student project demo, mention it in your report if
  markers test it live.
- No GPU on the free tier — inference is CPU-only, which is fine for single
  requests (a few seconds each) but not for high concurrency.
- If your `artifacts/` folder is large, consider a private Space (still
  free) so your trained weights aren't public.

**Alternative if you want it to never sleep:** Render.com's free web service
tier (also sleeps) or Railway's free trial credits — same Dockerfile works
unmodified on both; just set `PORT` via their dashboard env vars if they
require a different port than 7860 (the Dockerfile already reads `$PORT`).

---

## 4. Wiring your mobile app (React Native / Flutter) to this API

Call `POST /diagnose` as `multipart/form-data` with:
- `image` — the file field (JPG/PNG/WEBP)
- `description` — text field, the user's free-text symptom description
- `skin_type` — optional text field

Example from a React Native fetch call:
```js
const form = new FormData();
form.append('image', { uri: photoUri, name: 'skin.jpg', type: 'image/jpeg' });
form.append('description', userDescription);
if (skinType) form.append('skin_type', skinType);

const res = await fetch(`${API_BASE}/diagnose`, { method: 'POST', body: form });
const diagnosis = await res.json();
```

The JSON response shape matches `app/schemas.py` → `DiagnosisResponse`,
mirroring exactly the structure your `diagnose()` function in the fusion
notebook already produces, plus a `recommendation` block.

---

## 5. What's genuinely new here vs. your notebooks

Your three notebooks stop at a working `diagnose()` function inside Colab.
This backend adds the parts needed to make it a real, callable app:
- A persistent server process that loads models **once** (not per-request)
- Input validation, file-size limits, proper HTTP error codes
- CORS so a mobile/web frontend on a different origin can call it
- The **Phase 4 recommendation engine** (`app/recommend.py` +
  `app/recommendations_db.json`) — rule-based, keyed on your 9 shared
  classes, giving causes / key ingredients / routine / red-flags per
  condition, exactly the "semantic mapping" approach discussed in Section
  2.4.1 of your contextual report. Feel free to expand the JSON with more
  detail/products — no code changes needed, it's pure data.
- A `/health` endpoint useful both for your own debugging and as evidence
  in your report/demo that each component loads and is wired correctly.

## 6. Known things to double check once you drop in your real artifacts

- Confirm `image_model_meta.json`'s `img_size` matches what's actually in
  the file you place (should be `[300, 300]` per your training notebook).
- Confirm the class order in `image_label_classes.json` and
  `text_label_classes.json` are identical — the fusion notebook already
  asserts this at load time in Colab; this backend does not re-assert it,
  so a manual sanity check via `/health`'s `classes` field is worthwhile.
- `requirements.txt` pins `tensorflow-cpu==2.20.0` / `transformers==4.44.2`
  to match your training environment. If your Colab used different exact
  versions, align these to avoid `.keras` load warnings or state_dict
  mismatches.
