import pandas as pd

df = pd.read_csv("gdelt_2020_news.csv")

ts_utc = pd.to_datetime(df['PublishTimestamp'].str.replace(' UTC', ''), utc=True)
publish_local = ts_utc.dt.tz_convert('Europe/Athens').dt.tz_localize(None)
new_hours = (publish_local.dt.hour // 4) * 4
df['Candle4H_EET'] = publish_local.dt.normalize() + pd.to_timedelta(new_hours, unit='h')
df['PredictionCandle_EET'] = df['Candle4H_EET'] + pd.Timedelta(hours=4)
df['Candle4H'] = df['Candle4H_EET'].dt.strftime('%Y-%m-%d %H:%M:%S')
df['PredictionCandle'] = df['PredictionCandle_EET'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_ranked = df.copy()
df_ranked['AbsSentiment'] = df_ranked['SentimentScore'].abs()

df_selected = (
    df_ranked.sort_values(
        by=['Candle4H', 'MacroPriority', 'AbsSentiment', 'PublishTimestamp'],
        ascending=[True, False, False, False]
    )
    .groupby('Candle4H', sort=False)
    .head(20)
)
df_final = df_selected.sort_values(
    by=['Candle4H', 'PublishTimestamp'],
    ascending=[True, True]
)
df_final = df_final.drop(columns=['Candle4H_EET', 'PredictionCandle_EET', 'AbsSentiment'])

df_final.to_csv("gdelt_news_2020_eet_eest.csv", index=False)

old = pd.read_csv("gdelt_2020_news.csv")
new = pd.read_csv("gdelt_news_2020_eet_eest.csv")

print("Alte Anzahl:", len(old))
print("Neue Anzahl:", len(df_final))

lost = old[~old["URL"].isin(new["URL"])]
print(lost[[
    "PublishTimestamp",
    "Candle4H",
    "MacroPriority",
    "SentimentScore",
    "URL"
]])
lost["PublishTimestamp"] = pd.to_datetime(
    lost["PublishTimestamp"].str.replace(" UTC", ""),
    utc=True
)
print(lost["PublishTimestamp"].dt.month.value_counts().sort_index())
print(lost["MacroPriority"].value_counts())
print(old["MacroPriority"].value_counts())
before_counts = old.groupby("Candle4H").size()
after_counts = new.groupby("Candle4H").size()


def summarize_bucket_sizes(counts):
    return pd.Series({
        "1-5": ((counts >= 1) & (counts <= 5)).sum(),
        "6-10": ((counts >= 6) & (counts <= 10)).sum(),
        "11-15": ((counts >= 11) & (counts <= 15)).sum(),
        "16-20": ((counts >= 16) & (counts <= 20)).sum(),
        ">20": (counts > 20).sum(),
        "Max": counts.max(),
        "Durchschnitt": counts.mean()
    })

print("Vorher:")
print(summarize_bucket_sizes(before_counts))

print("\nNachher:")
print(summarize_bucket_sizes(after_counts))