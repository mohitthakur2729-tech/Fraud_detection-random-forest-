import numpy as np
import pandas as pd
import re
from app import app

# Get a known fraud row
df = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)
row = df.iloc[12]
print('Testing with fraud row 12:', row.values)

# Simulate form submission
with app.test_client() as client:
    # Build datetime string from row values
    dt_str = f"{int(row['trans_year'])}-{int(row['trans_month']):02d}-{int(row['trans_day']):02d}T{int(row['trans_hour']):02d}:00"
    # Build dob assuming age was computed as (trans_date - dob).days/365.25
    trans_dt = pd.Timestamp(f"{int(row['trans_year'])}-{int(row['trans_month']):02d}-{int(row['trans_day']):02d} {int(row['trans_hour']):02d}:00")
    dob = trans_dt - pd.Timedelta(days=int(row['age']*365.25))
    dob_str = dob.strftime('%Y-%m-%d')

    resp = client.post('/detect', data={
        'trans_datetime': dt_str,
        'category': str(int(row['category'])),
        'card_number': str(int(row['upi_number'])),
        'dob': dob_str,
        'trans_amount': str(row['trans_amount']),
        'state': str(int(row['state'])),
        'zip': str(int(row['zip'])),
    })
    print('Status:', resp.status_code)
    html = resp.data.decode('utf-8')
    match = re.search(r'<h1[^>]*id="output"[^>]*>(.*?)</h1>', html, re.S)
    output = match.group(1).strip() if match else ''
    print('Displayed output:', output)
    if output == 'FRAUD TRANSACTION':
        print('RESULT: Correctly predicted FRAUD')
    elif output == 'VALID TRANSACTION':
        print('RESULT: Predicted SAFE (BUG!)')
    else:
        print('Unexpected output')
        print(html[:500])
