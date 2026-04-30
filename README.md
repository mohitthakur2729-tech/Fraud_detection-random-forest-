# UPI Fraud Detection

## Abstract

UPI fraud detection is important because digital payment systems process many transactions quickly, making manual checking slow and unreliable. This project uses a supervised Random Forest classifier to classify each transaction as either **VALID TRANSACTION** or **FRAUD TRANSACTION**. The application converts user-entered transaction details into model features, including transaction time, category, UPI number, age, amount, state, and PIN code. A saved scaler normalizes these values, and the trained Random Forest model predicts fraud probability from learned transaction patterns. The system is designed as a student/demo Flask application that shows how machine learning can support fraud screening for both safe and suspicious UPI transactions.

## What This Project Does

This is a Flask web application that predicts whether a UPI transaction is a **VALID TRANSACTION** or a **FRAUD TRANSACTION**.

The project has two main user flows:

1. **Check**
   - User enters one transaction manually.
   - The backend converts the form values into model features.
   - The saved scaler and trained Random Forest model predict the result.

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
- `filesuse/random_forest_model.pkl` - Saved Random Forest fraud detection model.
- `filesuse/scaler.pkl` - Saved scikit-learn scaler used before model prediction.
- `dataset/upi_fraud_dataset.csv` - Dataset used for testing/reference.
- `train_random_forest.py` - Retrains the scaler and Random Forest model.

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

The scaler transforms the 10 features, then the Random Forest model predicts a fraud probability. A probability of `0.5` or higher is shown as **FRAUD TRANSACTION**. A probability below `0.5` is shown as **VALID TRANSACTION**.

Important: this is a student/demo fraud detector. It predicts from patterns in the local dataset and model artifacts. It does not verify real UPI IDs against a bank or payment network.

## Example Valid Test Input

Use this in the Check form to test the valid result:

- UPI number: `9957000001`
- UPI holder name: `Any Name`
- DOB: `1968-01-01`
- State: `Tamil Nadu`
- PIN/ZIP: `49879`
- Transaction date/time: `2022-01-01T00:00`
- Transaction amount: `66.21`
- Seller name: `Any Merchant`
- Category: `Shopping POS`

Expected result:

```text
VALID TRANSACTION
```

## Example Fraud Test Input

Use this in the Check form to test the fraud result:

- UPI number: `9957000013`
- UPI holder name: `Any Name`
- DOB: `1992-01-31`
- State: `West Bengal`
- PIN/ZIP: `28611`
- Transaction date/time: `2022-02-01T01:00`
- Transaction amount: `281.06`
- Seller name: `Any Merchant`
- Category: `Grocery POS`

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

Retrain the model:

```bash
python train_random_forest.py
```

Open:

```text
http://127.0.0.1:5000
```

## Run Checks

```bash
python -m py_compile app.py train_random_forest.py
```

## Current Limitations

- Uploaded CSV files are previewed only; they do not retrain the saved model.
- The saved model is only as reliable as the dataset it was trained on.
- UPI number, state, and PIN patterns may be dataset-specific, not real-world fraud indicators.
- For a production fraud system, you would need a larger real dataset, proper validation, model monitoring, explainability, and security controls.
