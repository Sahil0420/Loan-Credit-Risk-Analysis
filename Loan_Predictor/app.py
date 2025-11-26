from fastapi import FastAPI , Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
from pydantic import BaseModel
import joblib

model = joblib.load("../loan_default_model.pkl")

app = FastAPI(title="loan_default_predictor")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    person_age: int = Form(...),
    person_income: float = Form(...),
    person_emp_length: float = Form(...),
    loan_amnt: float = Form(...),
    loan_int_rate: float = Form(...),
    loan_percent_income: float = Form(...),
    person_home_ownership: str = Form(...),
    loan_intent: str = Form(...),
    loan_grade: str = Form(...),
    cb_person_default_on_file: str = Form(...)
):   

    data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file
    }

    df = pd.DataFrame([data])

    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)  # keep all categories

    for col in model.get_booster().feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[model.get_booster().feature_names]


    prob = model.predict_proba(df)[:,1][0]
    pred = "Default" if model.predict(df)[0] == 1 else "No Default"

    result = {
        "prediction": pred,
        "probability": round(prob*100, 2)
    }

    return templates.TemplateResponse("index.html", {"request": request, "result": result})
