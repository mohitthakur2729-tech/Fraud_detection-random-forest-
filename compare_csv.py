import pandas as pd

df1 = pd.read_csv('upi_fraud_dataset.csv', index_col=0)
df2 = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)

diff_mask = df1['upi_number'] != df2['upi_number']
diff_count = diff_mask.sum()
print('Different rows:', diff_count)
if diff_count > 0:
    idx = diff_mask[diff_mask].index[:10]
    print('Sample differences:')
    for i in idx:
        r1 = df1.loc[i, 'upi_number']
        r2 = df2.loc[i, 'upi_number']
        print(f'  Row {i}: root={r1}, dir={r2}, delta={r1-r2}')
