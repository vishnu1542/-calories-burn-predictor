from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import json
import numpy as np
import os

app = FastAPI(title="AI Calories Burned Predictor", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model and metadata
BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "calories_model.pkl"))
label_encoder = joblib.load(os.path.join(BASE, "label_encoder.pkl"))
with open(os.path.join(BASE, "metadata.json")) as f:
    metadata = json.load(f)

# ── Request schema ──────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    age: int
    gender: int          # 0 = Female, 1 = Male
    weight_kg: float
    height_cm: float
    body_fat_pct: float
    exercise_type: str
    duration_min: int
    intensity: int       # 1 = Low, 2 = Medium, 3 = High
    heart_rate: int

# ── Routes ───────────────────────────────────────────────────────────────────
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
        met_value = metadata["met_values"].get(data.exercise_type, 6.0)
        exercise_encoded = label_encoder.transform([data.exercise_type])[0]
        bmi = data.weight_kg / ((data.height_cm / 100) ** 2)

        features = np.array([[
            data.age,
            data.gender,
            data.weight_kg,
            data.height_cm,
            round(bmi, 2),
            data.body_fat_pct,
            exercise_encoded,
            met_value,
            data.duration_min,
            data.intensity,
            data.heart_rate
        ]])

        prediction = model.predict(features)[0]
        calories = round(float(prediction), 1)

        # Derived insights
        calories_per_min = round(calories / data.duration_min, 1)
        intensity_label = {1: "Low", 2: "Medium", 3: "High"}[data.intensity]
        
        # Equivalent food items (approx)
        food_equivalents = []
        if calories >= 50:
            food_equivalents.append({"item": "🍎 Apples", "count": round(calories / 52, 1)})
        if calories >= 250:
            food_equivalents.append({"item": "🍕 Pizza slices", "count": round(calories / 285, 1)})
        if calories >= 100:
            food_equivalents.append({"item": "🍫 Chocolate bars", "count": round(calories / 235, 1)})
        if calories >= 100:
            food_equivalents.append({"item": "🥤 Sodas (350ml)", "count": round(calories / 150, 1)})

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
