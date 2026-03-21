# 🔥 CalorieAI — Calories Burned Predictor

An AI-powered web app that predicts calories burned during workouts using a Gradient Boosting model trained on 10,000+ synthetic workout sessions.

## 📁 Project Structure

```
calories_app/
├── main.py                  # FastAPI application
├── generate_dataset.py      # Dataset generation script
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── model/
│   ├── calories_dataset.csv # Generated dataset (10,000 rows)
│   ├── calories_model.pkl   # Trained GBM model
│   ├── label_encoder.pkl    # Exercise type encoder
│   └── metadata.json        # Model metrics & exercise types
├── templates/
│   └── index.html           # Frontend UI (Jinja2)
└── static/                  # Static assets (CSS, JS, images)
```

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset (already done — skip if model files exist)
```bash
python generate_dataset.py
```

### 3. Train the model (already done — skip if .pkl files exist)
```bash
python train_model.py
```

### 4. Start the FastAPI server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open in browser
```
http://localhost:8000
```

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| R² Score | 0.9904 |
| MAE | ~33 kcal |
| RMSE | ~45 kcal |
| Algorithm | Gradient Boosting |
| Training samples | 10,000 |

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| POST | `/predict` | Calories prediction |
| GET | `/health` | API health check |
| GET | `/exercises` | List exercise types |

### POST `/predict` — Request Body
```json
{
  "age": 28,
  "gender": 0,
  "weight_kg": 65.0,
  "height_cm": 165.0,
  "body_fat_pct": 22.0,
  "exercise_type": "Running",
  "duration_min": 45,
  "intensity": 2,
  "heart_rate": 155
}
```

### Response
```json
{
  "success": true,
  "calories_burned": 487.3,
  "calories_per_min": 10.8,
  "bmi": 23.9,
  "met_value": 9.8,
  "intensity_label": "Medium",
  "food_equivalents": [
    {"item": "🍎 Apples", "count": 9.4},
    {"item": "🍕 Pizza slices", "count": 1.7}
  ],
  "exercise_type": "Running",
  "duration_min": 45
}
```

## 🧠 Model Input Features
| Feature | Description |
|---------|-------------|
| age | User age (15–75) |
| gender | 0=Female, 1=Male |
| weight_kg | Body weight in kg |
| height_cm | Height in cm |
| bmi | Auto-calculated |
| body_fat_pct | Body fat percentage |
| exercise_type | One of 10 exercise types |
| met_value | MET value (auto-mapped) |
| duration_min | Workout duration in minutes |
| intensity | 1=Low, 2=Medium, 3=High |
| heart_rate | Heart rate in BPM |

## 🏋️ Supported Exercise Types
Cycling, Elliptical, HIIT, Jump Rope, Rowing, Running, Swimming, Walking, Weight Training, Yoga

## 🔧 Tech Stack
- **Backend**: FastAPI + Python
- **ML Model**: Gradient Boosting Regressor (scikit-learn)
- **Frontend**: HTML/CSS/JS with Jinja2 templating
- **Data**: Synthetic dataset using MET + heart-rate formulas
