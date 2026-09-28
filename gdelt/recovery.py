import asyncio
import pandas as pd
import json
import logging
from openai import AsyncOpenAI
from datetime import timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

client = AsyncOpenAI(
    base_url="https://interweb.l3s.uni-hannover.de",
    api_key=os.getenv("L3S_API_KEY"),
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_top_themes(themes_str: str, top_n: int = 5) -> str:
    if not isinstance(themes_str, str) or not themes_str.strip():
        return ""
    raw_themes = [item.split(',')[0] for item in themes_str.split(';') if item]
    unique_themes = list(dict.fromkeys(raw_themes))
    top_themes = unique_themes[:top_n] 
    return ", ".join(top_themes)

MODEL_NAME = "gemma4:31b"
INPUT_CSV_PATH = "gdelt_news_2020_eet_eest.csv"
OUTPUT_JSON_PATH = "processed_macro_signals_2020_eet_eest.jsonl"

SYSTEM_PROMPT = """
You are a professional macroeconomic analyst specializing in foreign exchange markets.
You are given all relevant macroeconomic news published before the opening of the next 4-hour EUR/USD candle.
Your task is to extract the important macroeconomic information from the supplied news. Merge duplicate articles describing the same event and summarize the overall macroeconomic picture.
Do NOT make a trading recommendation or predict EUR/USD.
Return ONLY valid JSON with the following structure:
{
  "summary": "...",
  "macro_drivers": [
    {
      "topic": "...",
      "direction": "...",
      "strength": 0.0,
      "reason": "..."
    }
  ],
  "risks": [
    "..."
  ],
  "importance": 0.0,
  "confidence": 0.0
}
Guidelines:
- Use only the supplied news. Do not invent facts.
- Focus only on macroeconomic information relevant for financial markets.
- Merge duplicate articles referring to the same macroeconomic event.
- If multiple news outlets report the same event, treat it as a single event rather than increasing its importance.
- Extract the underlying macroeconomic implications instead of repeating headlines.
- Summarize the overall macroeconomic situation in 2-4 sentences.
- List at most the 5 most important macro drivers.
Macro driver format:
- topic: Use concise topic names whenever possible, such as:
  Federal Reserve, ECB, Interest Rates, Inflation, Employment,
  Economic Growth, Banking Sector, Government Debt, Trade,
  Energy, Oil, Geopolitics, Risk Sentiment, Financial Markets, Other.
- direction: Must be exactly one of: "hawkish", "dovish", "improving", "worsening", "escalating", "easing", "neutral".
- strength: A value between 0.0 and 1.0 indicating how strongly this driver is supported by the supplied news.
- reason: One concise sentence explaining why this driver was identified.
Additional fields:
- risks: List the major macroeconomic risks or uncertainties mentioned in the news. Return an empty array if none are identified.
- importance: A value between 0.0 and 1.0 indicating the expected overall market relevance of all supplied news over the next 24 hours.
  0.0 = no relevant macro information
  0.5 = normal scheduled macro developments
  1.0 = major market-moving event (e.g. Fed/ECB decision or systemic shock)
- confidence: A value between 0.0 and 1.0 indicating how consistently the supplied news supports the extracted macroeconomic picture.
Empty input handling:
If the supplied news is empty or contains no market-relevant macroeconomic information, return:
- an empty string for "summary",
- empty arrays for "macro_drivers" and "risks",
- importance = 0.0,
- confidence = 0.0.
Output ONLY valid JSON.
Do not include Markdown, code fences, explanations or additional text.
"""

async def process_candle_news(prediction_candle: str, group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Sendet die gesammelten News einer 4H-Kerze an das LLM und gibt das Ergebnis zurück.
    """
    if group_df.empty:
        return {"prediction_candle": prediction_candle, "llm_analysis": None}

    news_lines = []
    for idx, (_, row) in enumerate(group_df.iterrows(),1):
        outlet = row['NewsOutlet']
        sentiment = row['SentimentScore']
        themes = str(row['V2Themes_Clean'])
        orgs = str(row['V2Organizations_Clean'])
        persons = str(row['V2Persons_Clean'])
        
        news_item = f"""<news_item id="{idx}">Source: {outlet}\nGDELT Sentiment: {sentiment}\nEntities: {orgs}\nThemes: {themes}\nPersons: {persons}\nURL: {row['URL']}</news_item>"""
        news_lines.append(news_item)
    
    news_data_str = "\n\n---\n\n".join(news_lines)
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"News for candle {prediction_candle}:\n\n{news_data_str}"},
            ],
        )
        
        llm_output = response.choices[0].message.content.strip()
        if llm_output.startswith("```"):
            llm_output = llm_output.strip("`").replace("json\n", "", 1).strip()
        
        try:
            parsed_analysis = json.loads(llm_output)
            logging.info(f"✅ ERFOLG: Kerze {prediction_candle} erfolgreich verarbeitet.")
        except json.JSONDecodeError:
            logging.warning(f"⚠️ JSON-PARSE-FEHLER bei Kerze {prediction_candle}. Raw output gesichert.")
            parsed_analysis = {"raw_output": llm_output, "error": "JSON parse failed"}

        return {
            "prediction_candle": prediction_candle,
            "llm_analysis": parsed_analysis
        }

    except Exception as e:
        logging.error(f"Error processing candle {prediction_candle}: {str(e)}")
        return {"prediction_candle": prediction_candle, "llm_analysis": {"error": str(e)}}

async def run_manual_retries():
    # 1. HIER DEINE FEHLGESCHLAGENEN DATEN EINTRAGEN
    manual_failed_candles = [
        "2020-10-24 00:00:00",
        # "YYYY-MM-DD HH:MM:SS", ...
    ]

    if not manual_failed_candles:
        logging.info("Keine fehlgeschlagenen Kerzen im Array. Nichts zu tun.")
        return

    logging.info(f"Lade CSV-Daten für {len(manual_failed_candles)} manuelle Retries...")
    df = pd.read_csv(INPUT_CSV_PATH)
    df['PredictionCandle'] = pd.to_datetime(df['PredictionCandle'])
    
    # Gleiches Preprocessing wie im Hauptskript
    df['V2Themes_Clean'] = df['V2Themes'].apply(lambda x: extract_top_themes(x, top_n=5))
    df['V2Organizations_Clean'] = df['V2Organizations'].apply(lambda x: extract_top_themes(x, top_n=3))
    df['V2Persons_Clean'] = df['V2Persons'].apply(lambda x: extract_top_themes(x, top_n=3))
    
    grouped = df.groupby('PredictionCandle')

    # Lock für das sichere Schreiben in die bestehende Datei
    file_lock = asyncio.Lock()
    # Da es nur wenige sind, reicht hier ein kleines Limit
    semaphore = asyncio.Semaphore(10) 

    async def bound_process(candle_str, group):
        async with semaphore:
            result = await process_candle_news(candle_str, group)
        # Direktes Anhängen an die BESTEHENDE JSONL-Datei
        async with file_lock:
            with open(OUTPUT_JSON_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    tasks = []
    for candle_str in manual_failed_candles:
        candle_ts = pd.to_datetime(candle_str)
        if candle_ts in grouped.groups:
            group_df = grouped.get_group(candle_ts)
            tasks.append(bound_process(candle_str, group_df))
        else:
            logging.warning(f"⚠️ Keine News in der CSV für {candle_str} gefunden!")

    logging.info("Starte API-Anfragen für die fehlgeschlagenen Kerzen...")
    await asyncio.gather(*tasks)
    logging.info("✅ Manuelle Retries abgeschlossen! Hänge an Datei an...")

    # 2. FINALE JSON ERSTELLEN UND DEDUPLIZIEREN
    logging.info("Lese JSONL ein, überschreibe Fehler und erstelle finale JSON-Datei...")
    final_results_dict = {}
    
    with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    res = json.loads(line)
                    # Der Trick: Die neuen, erfolgreichen Einträge stehen ganz unten in der Datei.
                    # Dadurch überschreiben sie hier im Dictionary automatisch die alten Fehler.
                    final_results_dict[res['prediction_candle']] = res
                except json.JSONDecodeError:
                    continue

    all_results = list(final_results_dict.values())
    all_results.sort(key=lambda x: x['prediction_candle'])

    FINAL_JSON_PATH = "final_macro_signals_2020_eet_eest.json"
    with open(FINAL_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    logging.info(f"🎉 FERTIG! Finale und reparierte Daten gespeichert unter: {FINAL_JSON_PATH}")


if __name__ == "__main__":
    try:
        asyncio.run(run_manual_retries())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_manual_retries())