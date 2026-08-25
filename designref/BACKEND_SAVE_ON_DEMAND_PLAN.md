# Smart Dermatology Backend — Save-on-Demand History Fix
**Repo:** `smart_dermatology_backend`
**Covers:** stopping the automatic MongoDB write inside `/diagnose`, and adding a new endpoint the frontend will call only when the user explicitly chooses to save a result.
**Execution mode:** PHASED. Read this entire file first. Then execute exactly one phase at a time. After finishing a phase, STOP, report what changed, and wait for explicit approval before starting the next phase.

---

## Global rules

1. Do not touch `app/ml/*`, `app/recommend.py`, `app/concern_matcher.py`, `app/database.py`, `app/users.py`, `app/security.py`, `app/deps.py`, `app/auth_routes.py`, or `Dockerfile`. This is a small, surgical change to `app/schemas.py` and `app/main.py` only.
2. Do not touch `app/history.py` — `save_diagnosis`, `list_diagnoses`, `get_diagnosis`, and `delete_diagnosis` are all already correct and already reused as-is by the new endpoint below.
3. If anything below conflicts with code you actually find in the repo, stop and ask — do not guess or improvise a fix.

---

## Verified current state

`/diagnose` in `app/main.py` currently calls `history.save_diagnosis(...)` automatically inside the endpoint itself, every single time a diagnosis runs — regardless of whether the user ever chooses to save it. The frontend's "Save to History" button today only writes to its own local device storage and never calls this backend at all. The fix moves the actual database write to happen only when the frontend explicitly asks for it, via a new endpoint.

---

# PHASE 1 — Add the request schema

**File to edit:** `app/schemas.py`

Add this new model, placed after `DiagnosisResponse` is already defined (since it references it) — don't touch anything else in the file:

```python
class SaveHistoryRequest(BaseModel):
    description: str
    skin_type: str | None = None
    response: DiagnosisResponse
```

**Acceptance criteria:** File still imports/parses correctly (`python -c "from app.schemas import SaveHistoryRequest"` succeeds with no errors).

**STOP HERE.** Report the diff. Do not start Phase 2 until told to continue.

---

# PHASE 2 — Remove the auto-save from `/diagnose`; add `POST /history`

**File to edit:** `app/main.py`

**Step 1.** Update the schemas import line at the top to include the new model:
```python
from app.schemas import DiagnosisResponse, HealthResponse, SaveHistoryRequest
```
(edit the existing import line — don't add a duplicate one)

**Step 2.** In the `/diagnose` endpoint, find and delete this entire block (it currently sits right before `return response`):
```python
saved_history_id = None
if database.is_connected():
    try:
        saved = await history.save_diagnosis(
            final_condition=final_condition,
            final_confidence=final_confidence,
            description=description,
            skin_type=skin_type,
            response_payload=response.model_dump(),
            user_id=str(current_user["_id"]),
        )
        saved_history_id = saved["id"]
    except Exception:
        logger.exception("Failed to save diagnosis to MongoDB")

response.history_id = saved_history_id
```
After deleting it, the function should simply end with `return response`. `response.history_id` will remain `None` via its existing schema default — that's fine, nothing depends on it being non-null anymore.

**Step 3.** Add this new endpoint, placed near the other `/history*` routes:
```python
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
```

**Acceptance criteria — test with curl (or `static/test_client.html`) in this order, with the server running:**
1. Log in (or use an existing token) and call `POST /diagnose` with a real image + description → still returns a full result as before.
2. Immediately check MongoDB's `diagnoses` collection (or `GET /history` with that token) → **nothing new appears** from that call alone. This is the key regression check — if a new record shows up here, Step 2 above wasn't applied correctly.
3. Call `POST /history` with that same token and a body shaped like:
   ```json
   {
     "description": "test description",
     "skin_type": "oily",
     "response": { ...paste the full DiagnosisResponse JSON you got back from step 1... }
   }
   ```
   → expect `201` with a saved document that includes a Mongo `id`.
4. `GET /history` with that token → the record from step 3 now appears.
5. `GET /history`, `GET /history/{id}`, `DELETE /history/{id}` all still behave exactly as before for scoping/ownership (unchanged — you didn't touch these routes).

**STOP HERE.** This is the last phase for this repo — report the diff and the curl test results.
