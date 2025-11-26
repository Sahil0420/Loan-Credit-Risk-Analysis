import pandas as pd
import joblib

# Load model
model = joblib.load("../loan_default_model.pkl")

# Example input
data = {
    "person_age": 21,
    "person_income": 10000,
    "person_emp_length": 2,
    "loan_amnt": 50000,
    "loan_int_rate": 10,
    "loan_percent_income": 50000 / 10000,  # 0.5
    "person_home_ownership": "OWN",
    "loan_intent": "PERSONAL",
    "loan_grade": "A",
    "cb_person_default_on_file": "N"
}

# Create DataFrame
df = pd.DataFrame([data])

# One-hot encode like training
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)  # Keep all categories

# Align features with training
for col in model.get_booster().feature_names:
    if col not in df.columns:
        df[col] = 0

df = df[model.get_booster().feature_names]

# Predict
prob_default = model.predict_proba(df)[:,1][0]
pred_class = model.predict(df)[0]

print("Prediction:", "Default" if pred_class==1 else "No Default")
print("Probability of Default:", round(prob_default*100, 2), "%")
