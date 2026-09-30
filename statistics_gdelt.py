from pathlib import Path
from collections import Counter
import random

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_FOLDER = Path("analysis")
OUTPUT_FOLDER.mkdir(exist_ok=True)

df = pd.read_csv('gdelt/gdelt_news_2020_2026_eet_eest.csv')
print()
print(f"Total news: {len(df):,}")

df["PublishTimestamp"] = pd.to_datetime(df["PublishTimestamp"])
df["Candle4H"] = pd.to_datetime(df["Candle4H"])
df["PredictionCandle"] = pd.to_datetime(df["PredictionCandle"])
print("\n========================")
print("GENERAL")
print("========================")
print(df.describe(include="all"))

print("\n========================")
print("NEWS PER YEAR")
print("========================")

news_per_year = df.groupby(df.PublishTimestamp.dt.year).size()
print(news_per_year)
news_per_year.to_csv(OUTPUT_FOLDER / "news_per_year.csv")

print("\n========================")
print("NEWS PER SOURCE")
print("========================")

sources = df["NewsOutlet"].value_counts()
print(sources)
sources.to_csv(OUTPUT_FOLDER / "news_per_source.csv")

print("\n========================")
print("COVERAGE")
print("========================")

news_per_candle = df.groupby("Candle4H").size()

coverage = len(news_per_candle)

print(f"Candles containing news : {coverage}")
print(f"Average news/candle      : {news_per_candle.mean():.2f}")
print(f"Median news/candle       : {news_per_candle.median():.2f}")
print(f"Maximum news/candle      : {news_per_candle.max()}")

plt.figure(figsize=(10,6))
plt.hist(news_per_candle, bins=20)
plt.title("News per 4H Candle")
plt.xlabel("Number of News")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "hist_news_per_candle.png")

plt.figure(figsize=(10,6))
plt.hist(df["SentimentScore"], bins=40)
plt.title("Sentiment Score Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "hist_sentiment.png")

print("\n========================")
print("TOP THEMES")
print("========================")

theme_counter = Counter()
for themes in df["V2Themes"].dropna():
    for theme in str(themes).split(";"):
        theme_counter[theme.strip()] += 1

theme_df = pd.DataFrame(theme_counter.items(),columns=["Theme", "Count"]).sort_values("Count",ascending=False)
print(theme_df.head(50))
theme_df.to_csv(OUTPUT_FOLDER / "top_themes.csv",index=False)

print("\n========================")
print("TOP ORGANIZATIONS")
print("========================")

org_counter = Counter()
for orgs in df["V2Organizations"].dropna():
    for org in str(orgs).split(";"):
        org_counter[org.strip()] += 1

org_df = pd.DataFrame(org_counter.items(),columns=["Organization","Count"]).sort_values("Count",ascending=False)
print(org_df.head(50))
org_df.to_csv(OUTPUT_FOLDER / "top_organizations.csv",index=False)

print("\n========================")
print("TOP PERSONS")
print("========================")

person_counter = Counter()
for persons in df["V2Persons"].dropna():
    for person in str(persons).split(";"):
        person_counter[person.strip()] += 1

person_df = pd.DataFrame(person_counter.items(),columns=["Person","Count"]).sort_values("Count",ascending=False)
print(person_df.head(50))
person_df.to_csv(OUTPUT_FOLDER / "top_persons.csv",index=False)

print("\n========================")
print("TOP DAYS")
print("========================")

top_days = (df.groupby(df.PublishTimestamp.dt.date).size().sort_values(ascending=False))
print(top_days.head(50))
top_days.head(50).to_csv(OUTPUT_FOLDER / "top_days.csv")

summary = pd.DataFrame({

    "Metric":[
        "Total News",
        "Unique Candles",
        "Average News/Candle",
        "Median News/Candle",
        "Maximum News/Candle",
        "Unique Sources",
        "Unique Themes"
    ],
    "Value":[
        len(df),
        news_per_candle.shape[0],
        round(news_per_candle.mean(),2),
        round(news_per_candle.median(),2),
        news_per_candle.max(),
        df.NewsOutlet.nunique(),
        len(theme_counter)
    ]
})
summary.to_csv(OUTPUT_FOLDER / "summary.csv",index=False)
