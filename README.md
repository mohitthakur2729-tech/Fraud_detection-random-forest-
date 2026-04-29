# UPI Fraud Detection

## What This Project Does

This is a Flask web application that predicts whether a UPI transaction is a **VALID TRANSACTION** or a **FRAUD TRANSACTION**.

The project has two main user flows:

1. **Check**
   - User enters one transaction manually.
   - The backend converts the form values into model features.
   - The saved scaler and trained TensorFlow model predict the result.
   - A small high-risk rule fallback is also used for obviously risky inputs.

2. **Upload**
   - User uploads a CSV transaction dataset.
   - The app previews the uploaded data in a table.
   - The current app does not retrain the model from the upload page; the train button is only a UI simulation.

## Main Files

- `app.py` - Main Flask backend. It loads the model, handles routes, builds prediction features, and returns result pages.
- `templates/index.html` - Check form where a user enters transaction details.
- `templates/result.html` - Result screen shown after prediction.
- `templates/upload.html` - CSV upload page.
- `templates/preview.html` - Uploaded CSV preview page.
- `templates/chart.html` - Chart/dashboard page.
- `filesuse/project_model1.h5` - Saved TensorFlow fraud detection model.
- `filesuse/scaler.pkl` - Saved scikit-learn scaler used before model prediction.
- `dataset/upi_fraud_dataset.csv` - Dataset used for testing/reference.
- `src/build_model.ipynb` - Notebook used to build/train the original model.
- `test_prediction.py` - Tests saved model predictions against dataset rows.
- `test_web_predict.py` - Tests the Flask `/detect` route with a known fraud row.

## Check Section Inputs

The Check form collects:

1. UPI number
2. Date of birth
3. State
4. PIN/ZIP code
5. Transaction date and time
6. Transaction amount
7. Merchant category

The backend converts these into 10 model features:

1. Transaction hour
2. Transaction day
3. Transaction month
4. Transaction year
5. Category code
6. UPI number
7. Age derived from DOB and transaction date
8. Transaction amount
9. State code
10. PIN/ZIP code

## How Fraud Is Decided

The app uses:

1. **Model probability**
   - The scaler transforms the 10 features.
   - The TensorFlow model predicts a probability.
   - Probability above `0.5` is treated as fraud.

2. **High-risk fallback score**
   - Very high transaction amount
   - Late-night transaction time
   - Higher-risk category
   - Unusual age

If either the model predicts fraud or the fallback score is high, the app shows **FRAUD TRANSACTION**.

Important: this is a student/demo fraud detector. It predicts from patterns in the local dataset and model artifacts. It does not verify real UPI IDs against a bank or payment network.

## Example Fraud Test Input

Use this in the Check form to test the fraud result:

- UPI number: `988376137288`
- DOB: `1972-12-15`
- State: `Arunachal Pradesh`
- PIN/ZIP: `588317`
- Transaction date/time: `2020-12-27T18:30`
- Transaction amount: `88125`
- Category: `Travel`

Expected result:

```text
FRAUD TRANSACTION
```

## Run The Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Run Checks

```bash
python -m py_compile app.py test_web_predict.py
python test_prediction.py
python test_web_predict.py
```

## Current Limitations

- Uploaded CSV files are previewed only; they do not retrain the saved model.
- The saved model is only as reliable as the dataset it was trained on.
- UPI number, state, and PIN patterns may be dataset-specific, not real-world fraud indicators.
- For a production fraud system, you would need a larger real dataset, proper validation, model monitoring, explainability, and security controls.
