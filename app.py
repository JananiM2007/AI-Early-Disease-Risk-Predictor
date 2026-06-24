from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --------------------
# LOAD MODELS
# --------------------

diabetes_model = joblib.load(
    'diabetes_model.pkl'
)

diabetes_features = joblib.load(
    'diabetes_feature_names.pkl'
)

heart_model = joblib.load(
    'heart_model.pkl'
)

heart_scaler = joblib.load(
    'heart_scaler.pkl'
)

heart_features = joblib.load(
    'heart_feature_names.pkl'
)

kidney_model = joblib.load(
    'kidney_model.pkl'
)

kidney_features = joblib.load(
    'kidney_feature_names.pkl'
)


# --------------------
# HOME PAGE
# --------------------

@app.route('/')
def home():
    return render_template('index.html')


# --------------------
# DIABETES PAGE
# --------------------

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    age = float(request.form['Age'])
    pregnancies = float(request.form['Pregnancies'])
    glucose = float(request.form['Glucose'])
    bp = float(request.form['BloodPressure'])
    skin = float(request.form['SkinThickness'])
    insulin = float(request.form['Insulin'])

    height = float(request.form['Height'])
    weight = float(request.form['Weight'])

    family = float(request.form['FamilyHistory'])

    height_m = height / 100

    bmi = weight / (height_m ** 2)

    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3

    if age < 30:
        age_group = 0
    elif age < 50:
        age_group = 1
    else:
        age_group = 2

    high_glucose = 1 if glucose >= 140 else 0

    obese = 1 if bmi >= 30 else 0

    dpf = 0.8 if family == 1 else 0.3

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'BMI_Category': bmi_category,
        'Age_Group': age_group,
        'High_Glucose': high_glucose,
        'Obese': obese
    }
    #reasons
    reasons = []

    if glucose >= 140:
        reasons.append("High blood glucose")

    if bmi >= 30:
        reasons.append("Obesity")

    if age >= 45:
        reasons.append("Higher age risk")

    if family == 1:
        reasons.append("Family history of diabetes")

    df = pd.DataFrame([data])

    df = df[diabetes_features]

    probability = diabetes_model.predict_proba(df)[0][1]

    prediction = diabetes_model.predict(df)[0]

    return render_template(
        'result.html',
        disease='Diabetes',
        prediction=prediction,
        probability=round(probability*100,2),
        reasons=reasons
    )


# --------------------
# HEART PAGE
# --------------------

@app.route('/heart_quick')
def heart_quick():
    return render_template('heart_quick.html')

@app.route('/quick_heart_predict', methods=['POST'])
def quick_heart_predict():

    age = int(request.form['age'])
    smoking = int(request.form['smoking'])
    diabetes = int(request.form['diabetes'])
    bp = int(request.form['bp'])
    pain = int(request.form['pain'])
    exercise = int(request.form['exercise'])

    score = 0

    if age >= 50:
        score += 20

    if smoking:
        score += 20

    if diabetes:
        score += 20

    if bp:
        score += 20

    if pain:
        score += 15

    if exercise == 2:
        score += 15

    probability = min(score,100)

    reasons = []

    if smoking:
        reasons.append("Smoking increases heart disease risk")

    if bp:
        reasons.append("High blood pressure detected")

    if diabetes:
        reasons.append("Diabetes is a cardiovascular risk factor")

    if pain:
        reasons.append("Chest pain symptoms reported")

    if exercise == 2:
        reasons.append("Low physical activity")

    return render_template(
        'result.html',
        disease='Heart Disease',
        probability=probability,
        reasons=reasons
    )
@app.route('/heart')
def heart():
    return render_template('heart.html')



@app.route('/predict_heart', methods=['POST'])
def predict_heart():

    data = {
        'Age': float(request.form['Age']),
        'RestingBP': float(request.form['RestingBP']),
        'Cholesterol': float(request.form['Cholesterol']),
        'FastingBS': float(request.form['FastingBS']),
        'MaxHR': float(request.form['MaxHR']),
        'Oldpeak': float(request.form['Oldpeak']),
        'Sex_M': float(request.form['Sex_M']),
        'ChestPainType_ATA': float(request.form['ChestPainType_ATA']),
        'ChestPainType_NAP': float(request.form['ChestPainType_NAP']),
        'ChestPainType_TA': float(request.form['ChestPainType_TA']),
        'RestingECG_Normal': float(request.form['RestingECG_Normal']),
        'RestingECG_ST': float(request.form['RestingECG_ST']),
        'ExerciseAngina_Y': float(request.form['ExerciseAngina_Y']),
        'ST_Slope_Flat': float(request.form['ST_Slope_Flat']),
        'ST_Slope_Up': float(request.form['ST_Slope_Up'])
    }
    reasons = []

    if data['Cholesterol'] > 240:
        reasons.append("High cholesterol level")

    if data['RestingBP'] > 140:
        reasons.append("High blood pressure")

    if data['ExerciseAngina_Y'] == 1:
        reasons.append("Exercise-induced chest pain")

    if data['Oldpeak'] > 2:
        reasons.append("Abnormal ECG stress response")

    if data['FastingBS'] == 1:
        reasons.append("Elevated fasting blood sugar")

    df = pd.DataFrame([data])
    df = df[heart_features]

    df[[
        'Age',
        'RestingBP',
        'Cholesterol',
        'MaxHR',
        'Oldpeak'
    ]] = heart_scaler.transform(
        df[[
            'Age',
            'RestingBP',
            'Cholesterol',
            'MaxHR',
            'Oldpeak'
        ]]
    )
    probability = heart_model.predict_proba(df)[0][1]

    prediction = heart_model.predict(df)[0]

    return render_template(
        'result.html',
        disease='Heart Disease',
        prediction=prediction,
        probability=round(probability*100,2),
        reasons=reasons
    )


# --------------------
# KIDNEY PAGE
# --------------------

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')


@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():

    data = {}

    for feature in kidney_features:
        data[feature] = float(request.form[feature])
    reasons = []

    if data['sc'] > 1.2:
        reasons.append("High serum creatinine level")

    if data['bu'] > 40:
        reasons.append("Elevated blood urea")

    if data['htn'] == 1:
        reasons.append("Hypertension detected")

    if data['dm'] == 1:
        reasons.append("Diabetes mellitus present")

    if data['hemo'] < 12:
        reasons.append("Low hemoglobin level")

    if data['ane'] == 1:
        reasons.append("Anemia detected")
    df = pd.DataFrame([data])

    probability = kidney_model.predict_proba(df)[0][1]

    prediction = kidney_model.predict(df)[0]

    return render_template(
        'result.html',
        disease='Kidney Disease',
        prediction=prediction,
        probability=round(probability*100,2),
        reasons=reasons
    )


if __name__ == '__main__':
    app.run(debug=True)