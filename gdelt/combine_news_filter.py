import pandas as pd

# 1. CSV einlesen
df1 = pd.read_csv("gdelt_news_2020_eet_eest.csv")
df2 = pd.read_csv("gdelt_news_2021_eet_eest.csv")
df3 = pd.read_csv("gdelt_news_2022_eet_eest.csv")
df4 = pd.read_csv("gdelt_news_2023_eet_eest.csv")
df5 = pd.read_csv("gdelt_news_2024_eet_eest.csv")
df6 = pd.read_csv("gdelt_news_2025_eet_eest.csv")
df7 = pd.read_csv("gdelt_news_2026_until_June_1_eet_eest.csv")
df = pd.concat([df1,df2,df3,df4,df5,df6,df7], ignore_index=True);

df['PublishTimestamp'] = pd.to_datetime(df['PublishTimestamp'].str.replace(' UTC', ''), utc=True)

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
df_final = df_final.drop(columns=['AbsSentiment'])
df_final['PublishTimestamp'] = df_final['PublishTimestamp'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
df_final.to_csv("gdelt_news_2020_2026_eet_eest.csv", index=False)
print("Umgruppierung erfolgreich!")
old = df.copy()

new = pd.read_csv("gdelt_news_2020_2026_eet_eest.csv")

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