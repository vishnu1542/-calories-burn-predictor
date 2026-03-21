import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load dataset
df = pd.read_csv('calories_dataset.csv')

# Encode exercise type
le = LabelEncoder()
df['exercise_encoded'] = le.fit_transform(df['exercise_type'])

# Features & target
features = ['age', 'gender', 'weight_kg', 'height_cm', 'bmi',
            'body_fat_pct', 'exercise_encoded', 'met_value',
            'duration_min', 'intensity', 'heart_rate']
target = 'calories_burned'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE:  {mae:.2f} calories")
print(f"RMSE: {rmse:.2f} calories")
print(f"R²:   {r2:.4f}")

# Feature importance
importance = dict(zip(features, model.feature_importances_))
importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
print("\nFeature Importances:")
for k, v in importance_sorted.items():
    print(f"  {k}: {v:.4f}")

# Save model and encoder
joblib.dump(model, 'calories_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Save metadata
exercise_types = list(le.classes_)
met_values = {
    'Running': 9.8, 'Cycling': 7.5, 'Swimming': 8.0,
    'Weight Training': 5.0, 'Yoga': 3.0, 'HIIT': 12.0,
    'Walking': 3.5, 'Jump Rope': 11.0, 'Rowing': 8.5, 'Elliptical': 6.0
}
metadata = {
    'exercise_types': exercise_types,
    'met_values': met_values,
    'features': features,
    'metrics': {'mae': round(mae, 2), 'r2': round(r2, 4), 'rmse': round(rmse, 2)}
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Model saved successfully!")
print(f"Exercise types: {exercise_types}")
