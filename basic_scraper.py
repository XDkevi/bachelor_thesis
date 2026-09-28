import os
import re
import random
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm

BASE_URL = "https://www.federalreserve.gov"
YEARS = range(2020, 2027)
CSV_FILENAME = "fed_speeches_2020_2026.csv"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def clean_date(date_str):
    if not date_str:
        return ""
    date_str = " ".join(date_str.split())
    formats = [
        "%B %d, %Y",   
        "%m/%d/%Y",    
        "%m/%d/%y",   
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    print(f"Unbekanntes Datumsformat: {date_str}")
    return date_str

def collect_urls():
    speeches = []

    for year in YEARS:
        url = f"{BASE_URL}/newsevents/{year}-speeches.htm"
        r = session.get(url)
        r.encoding = "utf-8"
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        for event in soup.select(".eventlist__event"):
            parent = event.find_parent("div", class_="row")
            date_element = parent.select_one(".eventlist__time time")
            title_element = event.select_one("a[href*='/speech/']")
            speaker_element = event.select_one(".news__speaker")
            if title_element is None:
                continue
            speeches.append({
                "date": clean_date(date_element.get_text(strip=True)),
                "speaker": speaker_element.get_text(strip=True) if speaker_element else "",
                "title": title_element.get_text(strip=True),
                "url": BASE_URL + title_element["href"],
            })
    return pd.DataFrame(speeches)

def parse_speech(url):
    r = session.get(url, timeout=10)
    r.encoding = "utf-8"
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    article = soup.select_one("#article")
    if article is None:
        raise ValueError("No #article-Container found.")
    
    headline = ""
    headline_element = (
        article.select_one("h3.title")
        or article.select_one("h3")
        or article.select_one("h1")
    )
    if headline_element:
        headline = headline_element.get_text(" ", strip=True)
    time_str = ""
    meta_text = article.get_text("\n", strip=True)
    m = re.search(
        r"For release at\s+([0-9:apmAPM\s]+)",
        meta_text,
        re.IGNORECASE
    )
    if m:
        time_str = m.group(1).strip()
    content_div = None
    for div in article.find_all("div"):
        paragraphs = div.find_all("p", recursive=False)
        if len(paragraphs) >= 5:
            content_div = div
            break
    if content_div is None:
        content_div = article

    paragraphs = []
    for p in content_div.find_all("p", recursive=False):
        txt = p.get_text(" ", strip=True)
        if not txt:
            continue
        if txt == "Return to text":
            continue
        if txt == "Notes":
            break
        if txt.startswith("1. ") and "Return to text" in txt:
            break
        paragraphs.append(txt)

    text = "\n\n".join(paragraphs)
    return {
        "headline": headline,
        "time": time_str,
        "text": text
    }


def main():
    urls_df = collect_urls()
    print(f"Found {len(urls_df)} speeches.")
    scraped_urls = set()
    if os.path.exists(CSV_FILENAME):
        try:
            existing_df = pd.read_csv(CSV_FILENAME)
            scraped_urls = set(existing_df["url"].dropna().tolist())
            print(f"Continue: {len(scraped_urls)} speeches already in csv.")
        except Exception:
            print("CSV not readable.")

    todo_df = urls_df[~urls_df["url"].isin(scraped_urls)]
    print(f"{len(todo_df)} speeches need to be scraped.")

    if todo_df.empty:
        return

    for _, row in tqdm(todo_df.iterrows(), total=len(todo_df), desc="Scrape Details"):
        try:
            speech = parse_speech(row.url)
            if len(speech["text"]) < 500:
                print(f"\nShort text ({len(speech['text'])} characters) for:")
                print(row.url)

            new_row = {
                "date": row.date,
                "time": speech["time"],
                "speaker": row.speaker,
                "headline": speech["headline"],
                "maintext": speech["text"],
                "url": row.url
            }
            
            file_exists = os.path.exists(CSV_FILENAME)
            pd.DataFrame([new_row]).to_csv(
                CSV_FILENAME,
                mode="a",
                index=False,
                header=not file_exists,
                encoding="utf-8-sig"
            )

        except Exception as e:
            print(f"\nError: URL {row.url}: {e}")
        
        time.sleep(random.uniform(0.4, 2.5))


if __name__ == "__main__":
    main()