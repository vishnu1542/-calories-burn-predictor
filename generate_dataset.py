import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

# Personal features
age = np.random.randint(15, 75, n)
gender = np.random.choice([0, 1], n)  # 0=Female, 1=Male
weight = np.where(gender == 1,
                  np.random.normal(78, 12, n),
                  np.random.normal(63, 10, n))
height = np.where(gender == 1,
                  np.random.normal(175, 8, n),
                  np.random.normal(162, 7, n))
bmi = weight / ((height / 100) ** 2)
body_fat = np.where(gender == 1,
                    np.random.normal(18, 5, n),
                    np.random.normal(25, 5, n))

# Workout features
exercise_types = {
    'Running': 9.8,
    'Cycling': 7.5,
    'Swimming': 8.0,
    'Weight Training': 5.0,
    'Yoga': 3.0,
    'HIIT': 12.0,
    'Walking': 3.5,
    'Jump Rope': 11.0,
    'Rowing': 8.5,
    'Elliptical': 6.0
}

exercise_type = np.random.choice(list(exercise_types.keys()), n)
met_value = np.array([exercise_types[e] for e in exercise_type])
duration = np.random.randint(15, 120, n)
intensity = np.random.choice([1, 2, 3], n)  # 1=Low, 2=Medium, 3=High
heart_rate = (60 + (intensity * 30) + np.random.normal(0, 10, n)).clip(60, 200)

# Calories burned using MET + HR hybrid formula
calories_met = (met_value * intensity * weight * duration) / 60

# Heart rate formula (Keytel et al.)
calories_hr_male = duration * (0.6309 * heart_rate + 0.1988 * weight + 0.2017 * age - 55.0969) / 4.184
calories_hr_female = duration * (0.4472 * heart_rate - 0.1263 * weight + 0.074 * age - 20.4022) / 4.184
calories_hr = np.where(gender == 1, calories_hr_male, calories_hr_female)

# Weighted hybrid
calories_burned = (0.6 * calories_met + 0.4 * calories_hr)
calories_burned = np.clip(calories_burned + np.random.normal(0, 15, n), 30, 1500)

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'weight_kg': weight.round(1),
    'height_cm': height.round(1),
    'bmi': bmi.round(2),
    'body_fat_pct': body_fat.clip(5, 50).round(1),
    'exercise_type': exercise_type,
    'met_value': met_value,
    'duration_min': duration,
    'intensity': intensity,
    'heart_rate': heart_rate.round(0).astype(int),
    'calories_burned': calories_burned.round(1)
})

df.to_csv('calories_dataset.csv', index=False)
print(f"Dataset saved: {len(df)} rows")
print(df.describe())
