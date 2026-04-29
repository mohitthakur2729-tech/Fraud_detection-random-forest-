import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load dataset
df = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)
print('Dataset columns:', df.columns.tolist())

# Load scaler and model
scaler = joblib.load('filesuse/scaler.pkl')
model = tf.keras.models.load_model('filesuse/project_model1.h5')

# Get a known fraud row (row 12)
row = df.iloc[12]
print('Row 12 values:', row.values)
print('Row 12 fraud_risk:', row['fraud_risk'])

# Simulate app.py feature construction from form values
trans_datetime = pd.Timestamp(f"{int(row['trans_year'])}-{int(row['trans_month']):02d}-{int(row['trans_day']):02d} {int(row['trans_hour']):02d}:00")
dob = trans_datetime - pd.Timedelta(days=int(row['age']*365.25))

v1 = trans_datetime.hour
v2 = trans_datetime.day
v3 = trans_datetime.month
v4 = trans_datetime.year
v5 = int(row['category'])
v6 = float(row['upi_number'])
v7 = np.round((trans_datetime - dob).days / 365.25)
v8 = float(row['trans_amount'])
v9 = int(row['state'])
v10 = int(row['zip'])

x_test = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])
print('x_test from app.py logic:', x_test)
print('Original row features:', row.iloc[:10].values)

# Predict using app.py's exact method
y_pred = model.predict(scaler.transform([x_test]), verbose=0)
print('Prediction (app.py method):', y_pred[0][0])
print('Result:', 'FRAUD' if y_pred[0][0] > 0.5 else 'SAFE')

# Predict using raw row
y_pred2 = model.predict(scaler.transform([row.iloc[:10].values]), verbose=0)
print('Prediction (raw row):', y_pred2[0][0])
print('Result:', 'FRAUD' if y_pred2[0][0] > 0.5 else 'SAFE')

