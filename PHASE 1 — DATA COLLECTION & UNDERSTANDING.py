import pandas as pd

df = pd.read_csv("/mnt/data/S&P500.csv")

# Convert Date and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

print(df.head())
print(df.tail())
print(df.info())
print(df.isna().sum())
