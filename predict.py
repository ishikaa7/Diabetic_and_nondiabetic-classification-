import pandas as pd
import joblib
import numpy as np

print("🩺 Booting up Virtual Doctor AI...")

# 1. Load the Brain and the Translator
try:
    rf_model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ AI Brain and Z-Score Scaler loaded successfully!\n")
except FileNotFoundError:
    print("❌ ERROR: Could not find the .pkl files. Make sure they are in the same folder!")

# 2. Create a "New Patient" (You can change these numbers to test different people!)
# Let's test a high-risk patient: High BMI, older age, poor general health, high blood pressure.
new_patient_data = {
    'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'Smoker': 1, 'Stroke': 0,
    'HeartDiseaseorAttack': 0, 'PhysActivity': 0, 'Fruits': 0, 'Veggies': 1,
    'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0, 'Sex': 1, 'DiffWalk': 1,
    'GenHlth': 4, 'Age': 10, 'Education': 4, 'Income': 5, 'MentHlth': 15, 'PhysHlth': 20,
    'BMI': 38.5 
}

# 3. Format the patient data so the scaler can read it
# We MUST use the exact same column order we used during training
binary_cols = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
               'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
               'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'Sex', 'DiffWalk']
ordinal_cols = ['GenHlth', 'Age', 'Education', 'Income', 'MentHlth', 'PhysHlth']
continuous_cols = ['BMI']

feature_order = binary_cols + ordinal_cols + continuous_cols
patient_df = pd.DataFrame([new_patient_data], columns=feature_order)

# 4. Scale the Data (Translate it into AI Math)
scaled_patient_array = scaler.transform(patient_df)
scaled_patient_df = pd.DataFrame(scaled_patient_array, columns=feature_order)

# 5. Make the Diagnosis with the 0.45 Safety Threshold
confidence_percentage = rf_model.predict_proba(scaled_patient_df)[0][1]

print("=======================================")
print(f"📊 Patient Analysis Complete")
print(f"🔬 AI Confidence of Diabetes: {confidence_percentage * 100:.1f}%")
print("=======================================")

if confidence_percentage >= 0.45:
    print("🚨 DIAGNOSIS: HIGH RISK (DIABETIC/PREDIABETIC).")
    print("Action: Immediate blood panel and medical intervention required.")
else:
    print("✅ DIAGNOSIS: LOW RISK (HEALTHY).")
    print("Action: Standard preventative care.")