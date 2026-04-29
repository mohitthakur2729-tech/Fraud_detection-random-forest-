import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load scaler and model
scaler = joblib.load('filesuse/scaler.pkl')
model = tf.keras.models.load_model('filesuse/project_model1.h5')

# Test with a completely realistic in-range input
test_cases = [
    # (desc, trans_datetime_str, category, card_number, dob_str, trans_amount, state, zip)
    ("Typical safe-like", "2023-10-15T14:30", 11, 7500000000, "1993-05-21", 100.0, 13, 400001),
    ("High amount", "2023-10-15T14:30", 3, 7500000000, "1993-05-21", 3000.0, 13, 400001),
    ("Very low UPI", "2023-10-15T14:30", 11, 1234567890, "1993-05-21", 100.0, 13, 400001),
    ("High amount + odd hour", "2023-10-15T03:00", 3, 7500000000, "1993-05-21", 3000.0, 13, 400001),
    ("Young user", "2023-10-15T14:30", 11, 7500000000, "2005-05-21", 50.0, 13, 400001),
    ("Dataset-like fraud", "2022-02-01T01:00", 4, 9957000013, "1992-01-31", 281.06, 27, 28611),
    ("Dataset safe row 0", "2022-01-01T00:00", 12, 6900000000, "1968-01-01", 66.21, 22, 49879),
    ("High amount grocery", "2023-10-15T14:30", 4, 8000000000, "1980-01-01", 3000.0, 13, 400001),
    ("Late night misc", "2023-10-15T02:00", 8, 8000000000, "1980-01-01", 2000.0, 13, 400001),
]

for desc, dt_str, cat, card, dob_str, amt, st, zp in test_cases:
    trans_dt = pd.to_datetime(dt_str)
    dob = pd.to_datetime(dob_str)
    v1 = trans_dt.hour
    v2 = trans_dt.day
    v3 = trans_dt.month
    v4 = trans_dt.year
    v5 = cat
    v6 = float(card)
    v7 = np.round((trans_dt - dob).days / 365.25)
    v8 = float(amt)
    v9 = st
    v10 = zp
    x = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])
    y = model.predict(scaler.transform([x]), verbose=0)
    res = "FRAUD" if y[0][0] > 0.5 else "SAFE"
    print(f"{desc:30s} -> prob={y[0][0]:.6f}, result={res}")
