import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.exceptions import BadRequest

# Load the pre-fitted scaler (must match the one used during model training)
scaler = joblib.load('filesuse/scaler.pkl')

# Load the trained model
model = tf.keras.models.load_model('filesuse/project_model1.h5')

app = Flask(__name__)

FRAUD_THRESHOLD = 0.5


def build_features(form):
    required_fields = [
        "trans_datetime",
        "dob",
        "category",
        "card_number",
        "trans_amount",
        "state",
        "zip",
    ]
    missing_fields = [field for field in required_fields if not form.get(field)]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    trans_datetime = pd.to_datetime(form.get("trans_datetime"), errors="coerce")
    dob = pd.to_datetime(form.get("dob"), errors="coerce")
    if pd.isna(trans_datetime) or pd.isna(dob):
        raise ValueError("Enter valid transaction date/time and date of birth.")

    age = np.round((trans_datetime - dob).days / 365.25)
    if age < 0:
        raise ValueError("Date of birth cannot be after the transaction date.")

    return np.array([
        trans_datetime.hour,
        trans_datetime.day,
        trans_datetime.month,
        trans_datetime.year,
        int(form.get("category")),
        float(form.get("card_number")),
        age,
        float(form.get("trans_amount")),
        int(form.get("state")),
        int(form.get("zip")),
    ], dtype=float)


def high_risk_rule_score(features):
    trans_hour = features[0]
    category = features[4]
    age = features[6]
    trans_amount = features[7]

    score = 0
    if trans_amount >= 5000:
        score += 3
    elif trans_amount >= 1000:
        score += 1

    if trans_hour <= 4 or trans_hour >= 23:
        score += 1

    if category in {8, 9, 10, 11, 13}:
        score += 1

    if age < 18 or age > 85:
        score += 1

    return score


def predict_transaction(features):
    scaled_features = scaler.transform([features])
    model_probability = float(model.predict(scaled_features, verbose=0)[0][0])
    rule_score = high_risk_rule_score(features)

    is_fraud = model_probability > FRAUD_THRESHOLD or rule_score >= 3
    result = "FRAUD TRANSACTION" if is_fraud else "VALID TRANSACTION"
    return result, model_probability, rule_score


@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/home')
def home():
    return render_template('first.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview', methods=["POST"])
def preview():
    dataset = request.files.get('datasetfile')
    if not dataset or dataset.filename == '':
        raise BadRequest("Please upload a CSV file.")

    if not dataset.filename.lower().endswith('.csv'):
        raise BadRequest("Only CSV files can be previewed.")

    try:
        df = pd.read_csv(dataset, encoding='unicode_escape')
    except Exception as exc:
        raise BadRequest("Unable to read the uploaded CSV file.") from exc

    if 'Id' in df.columns:
        df.set_index('Id', inplace=True)

    return render_template("preview.html", df_view=df.head(100)) 


@app.route('/prediction1', methods=['GET'])
def prediction1():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        x_test = build_features(request.form)
        result, probability, rule_score = predict_transaction(x_test)
    except ValueError as exc:
        return render_template('result.html', OUTPUT=f"INVALID INPUT: {exc}"), 400

    app.logger.info(
        "Prediction: %s, model_probability=%.6f, rule_score=%s",
        result,
        probability,
        rule_score,
    )
    return render_template('result.html', OUTPUT='{}'.format(result))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port dynamically
    app.run(host="0.0.0.0", port=port)


