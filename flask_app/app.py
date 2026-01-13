from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load trained artifacts
# -----------------------------
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('rfe_model.pkl', 'rb') as f:
    rfe = pickle.load(f)

with open('stacking_regressor.pkl', 'rb') as f:
    model = pickle.load(f)

# -----------------------------
# Home route
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -----------------------------
# Prediction route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -------- Raw Inputs --------
        age = float(request.form['age'])
        height = float(request.form['height'])        # cm
        weight = float(request.form['weight'])        # kg
        duration = float(request.form['duration'])    # minutes
        heart_rate = float(request.form['heart_rate'])# bpm
        body_temp = float(request.form['body_temp'])  # °C
        gender = request.form['gender']               # male / female

        # -------- Feature Engineering (EXACTLY like notebook) --------

        # BMI
        bmi = weight / ((height / 100) ** 2)

        # Polynomial interaction features
        weight_duration = weight * duration
        weight_heart_rate = weight * heart_rate
        duration_heart_rate = duration * heart_rate

        # Age Group (same bins as training)
        if age <= 18:
            age_group = 'Youth'
        elif age <= 30:
            age_group = 'Young Adult'
        elif age <= 50:
            age_group = 'Middle-Aged'
        else:
            age_group = 'Senior'

        # -------- Create DataFrame in EXACT expected order --------
        input_df = pd.DataFrame([{
            'Heart_Rate': heart_rate,
            'Height': height,
            'Weight': weight,
            'Duration': duration,
            'Duration Heart_Rate': duration_heart_rate,
            'Age': age,
            'BMI': bmi,
            'Weight Heart_Rate': weight_heart_rate,
            'Weight Duration': weight_duration,
            'Body_Temp': body_temp,
            'Gender': gender,
            'Age_Group': age_group
        }])

        # -------- Preprocessing --------
        X_processed = preprocessor.transform(input_df)

        # -------- RFE Feature Selection --------
        X_selected = rfe.transform(X_processed)

        # -------- Prediction --------
        prediction = model.predict(X_selected)
        predicted_calories = round(prediction[0], 2)

        return render_template(
            'index.html',
            prediction_text=f'Predicted Calories Burned: {predicted_calories}'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error occurred: {e}'
        )

# -----------------------------
# Run Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
