import pandas as pd

df = pd.read_csv("EURUSD_D1_2010_2026.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])
start_date = pd.Timestamp("2020-01-01 00:00:00")
end_date = pd.Timestamp("2026-06-02 00:00:00")
mask = (df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)
filtered_df = df.loc[mask]
output_file = "valid_d1_dates_2020_2026.csv"
filtered_df[['DateTime']].to_csv(output_file, index=False)