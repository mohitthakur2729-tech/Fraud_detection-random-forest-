import numpy as np
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)
row = df.iloc[12]
print('Row 12:', row.values)

# Simulate app.py detect() logic EXACTLY as it would receive from form:
trans_datetime = pd.to_datetime("2022-02-01T01:00")  # reconstructed from row
v1 = trans_datetime.hour
v2 = trans_datetime.day
v3 = trans_datetime.month
v4 = trans_datetime.year
v5 = int(row['category'])
v6 = float(row['upi_number'])
dob = pd.to_datetime("1992-01-31")  # assuming age 30 from row
v7 = np.round((trans_datetime - dob).days / 365.25)
v8 = float(row['trans_amount'])
v9 = int(row['state'])
v10 = int(row['zip'])

x_test = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])
print('x_test from app.py arrange:', x_test)
print('Dataset row arrange:', row.iloc[:10].values)

# Compare values one by one
for i, (a, b) in enumerate(zip(x_test, row.iloc[:10].values)):
    print(f'Feature {i}: app={a}, dataset={b}, match={np.isclose(a,b)}')
