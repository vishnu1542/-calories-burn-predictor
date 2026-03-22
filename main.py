import os, sys, json, joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ── Paths ────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE, "model")
STATIC_DIR    = os.path.join(BASE, "static")
TEMPLATES_DIR = os.path.join(BASE, "templates")

# ── Auto-train if model missing ───────────────────────────────────────────────
sys.path.insert(0, BASE)   # ensures train_on_start.py is always found
try:
    from train_on_start import ensure_model_exists
    ensure_model_exists()
except Exception as e:
    print(f"❌ TRAINING ERROR: {e}", flush=True)
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model         = joblib.load(os.path.join(MODEL_DIR, "calories_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    print("✅ Model loaded successfully!", flush=True)
except Exception as e:
    print(f"❌ MODEL LOAD ERROR: {e}", flush=True)
    sys.exit(1)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="CalorieAI", version="1.0.0")

# Mount static only if folder exists and is not empty
if os.path.isdir(STATIC_DIR) and os.listdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ── Schema ────────────────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    age: int
    gender: int
    weight_kg: float
    height_cm: float
    body_fat_pct: float
    exercise_type: str
    duration_min: int
    intensity: int
    heart_rate: int

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "exercise_types": metadata["exercise_types"],
        "metrics": metadata["metrics"]
    })

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        met_value        = metadata["met_values"].get(data.exercise_type, 6.0)
        exercise_encoded = label_encoder.transform([data.exercise_type])[0]
        bmi              = data.weight_kg / ((data.height_cm / 100) ** 2)

        features = np.array([[
            data.age, data.gender, data.weight_kg, data.height_cm,
            round(bmi, 2), data.body_fat_pct, exercise_encoded,
            met_value, data.duration_min, data.intensity, data.heart_rate
        ]])

        calories         = round(float(model.predict(features)[0]), 1)
        calories_per_min = round(calories / data.duration_min, 1)
        intensity_label  = {1: "Low", 2: "Medium", 3: "High"}[data.intensity]

        food_equivalents = []
        if calories >= 50:  food_equivalents.append({"item": "🍎 Apples",         "count": round(calories/52,  1)})
        if calories >= 250: food_equivalents.append({"item": "🍕 Pizza slices",   "count": round(calories/285, 1)})
        if calories >= 100: food_equivalents.append({"item": "🍫 Chocolate bars", "count": round(calories/235, 1)})
        if calories >= 100: food_equivalents.append({"item": "🥤 Sodas (350ml)",  "count": round(calories/150, 1)})

        return JSONResponse({
            "success": True,
            "calories_burned": calories,
            "calories_per_min": calories_per_min,
            "bmi": round(bmi, 1),
            "met_value": met_value,
            "intensity_label": intensity_label,
            "food_equivalents": food_equivalents,
            "exercise_type": data.exercise_type,
            "duration_min": data.duration_min
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)

@app.get("/health")
async def health():
    return {"status": "ok", "model_r2": metadata["metrics"]["r2"]}

@app.get("/exercises")
async def get_exercises():
    return {"exercises": metadata["exercise_types"], "met_values": metadata["met_values"]}
