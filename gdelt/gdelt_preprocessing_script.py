import asyncio
import pandas as pd
import json
from openai import AsyncOpenAI
from datetime import timedelta
from typing import List, Dict, Any
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = AsyncOpenAI()
def extract_top_themes(themes_str: str, top_n: int = 5) -> str:
    if not isinstance(themes_str, str) or not themes_str.strip():
        return ""
    raw_themes = [item.split(',')[0] for item in themes_str.split(';') if item]
    unique_themes = list(dict.fromkeys(raw_themes))
    top_themes = unique_themes[:top_n] 
    return ", ".join(top_themes)

MODEL_NAME = "gemma4:31b"
INPUT_CSV_PATH = "news_data_multi_granularity.csv"
OUTPUT_JSON_PATH = "processed_macro_signals_2020_2026_eet_eest_new_new_12h.jsonl"

START_DATE = "2020-01-01"
END_DATE = "2026-06-02"
CANDLE_COL = "PredictionCandle_12H"

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
      "affected_currency": "...",
      "status": "...",
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
- Summarize the overall macroeconomic situation in MAX 2 sentences.
- List at most the 5 most important macro drivers.
- SEVERE FILTERING: Ignore general macroeconomic commentary, vague opinions, or standard daily market recaps (e.g., "Markets closed mixed today", "Investors await data").
- IDENTIFYING SURPRISES (LINGUISTIC CUES): Since you only see a 4-hour window, determine "surprises" or "shocks" based on the linguistic framing of the news. Look for keywords like "unexpected", "surprises markets", "beats estimates", "sudden", "escalates", or "historic".
- Events such as policy rate decisions, official CPI releases, major military escalations, or emergency financial interventions should normally contribute at least 0.5 to the overall importance score, even if they were widely expected.
- DOWNGRADE REITERATIONS: If the news explicitly uses phrasing that implies old information (e.g., "inflation remains a concern", "the Fed is still watching data", "as previously expected"), score importance as 0.0.
- Prefer the underlying macroeconomic cause over market reactions.
- Do not identify stock market movements, bond yields or exchange rate movements as macro drivers unless they are themselves the primary event.
- When multiple macro drivers are closely related, prefer fewer, broader drivers over many overlapping ones.

Macro driver format:
- topic: Must be exactly one of:
  Federal Reserve, ECB, Interest Rates, Inflation, Employment,
  Economic Growth, Banking Sector, Government Debt, Trade,
  Energy, Oil, Geopolitics, Risk Sentiment, Financial Markets, Other.
- direction: Must align logically with the chosen topic:
  * For "Federal Reserve", "ECB", "Interest Rates", or "Inflation": Use "hawkish", "dovish", or "neutral". (Note: Rising inflation is hawkish, falling inflation is dovish).
  * For "Employment", "Economic Growth", "Banking Sector", "Government Debt", "Trade", "Risk Sentiment", or "Financial Markets": Use "improving", "worsening", or "neutral".
  * For "Geopolitics", "Energy", or "Oil": Use "escalating", "easing", or "neutral".
  * For "Other": Choose the most logical direction from the options above.
- affected_currency: Must be exactly one of: "usd", "eur", "both", "global", "other".
- status: Must be exactly one of: "new" (just announced), "update" (ongoing development), "resolved" (ended/concluded).
- strength: A value between 0.0 and 1.0 indicating how strongly the supplied news supports the existence of this macro driver, independent of its market impact.
- reason: One concise sentence explaining why this driver was identified.
Additional fields:
- risks: List the major macroeconomic risks or uncertainties mentioned in the news (MAX 3). Return an empty array if none are identified.
- importance: A value between 0.0 and 1.0 indicating the expected overall market relevance for major FX markets, especially EUR/USD.
  0.0 = Background noise, general commentary, reiterated facts.
  0.5 = New data point or distinct shift in narrative.
  1.0 = Massive systemic shock (unexpected intervention, major escalation).
- confidence: A value between 0.0 and 1.0 indicating how accurately the extracted macro picture reflects the supplied information.
Empty input handling:
If the supplied news is empty or contains no market-relevant macroeconomic information, return:
- an empty string for "summary",
- empty arrays for "macro_drivers" and "risks",
- importance = 0.0,
- confidence = 0.0.
Output ONLY valid JSON.
Do not include Markdown, code fences, explanations or additional text.
"""

SYSTEM_PROMPT_NEW = """
You are a professional macroeconomic analyst specializing in foreign exchange markets.
You are given all relevant macroeconomic news published before the opening of the next 12-hour EUR/USD candle.
Your task is to extract the important macroeconomic information from the supplied news. Merge duplicate articles describing the same event and summarize the overall macroeconomic picture.
Do NOT make a trading recommendation or predict EUR/USD.
Return ONLY valid JSON with the following structure:
{
  "summary": "...",
  "macro_drivers": [
    {
      "topic": "...",
      "direction": "...",
      "affected_currency": "...",
      "status": "...",
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
- If several independent articles consistently support the same macroeconomic conclusion, increase confidence, but do not treat them as separate macro drivers.
- Extract the underlying macroeconomic implications instead of repeating headlines.
- Summarize the overall macroeconomic situation in MAX 2 sentences.
- List at most the 5 most important macro drivers.
- Filter out commentary that does not materially improve the understanding of the current macroeconomic environment.
- Evaluate if the news describes an ongoing macro trend (Context) or a breaking development (Trigger). Do NOT set importance to 0.0 simply because an ongoing trend is mentioned. Assign a moderate importance score for strong active macro regimes/confirmations, and a high score for breaking news, severe shocks, or surprises.
- Events such as policy rate decisions, official CPI releases, major military escalations or emergency financial interventions are generally of high macroeconomic relevance. Their importance should reflect both the significance of the event and whether it materially changes market expectations.
- Do not identify stock market movements, bond yields or exchange rate movements as macro drivers unless they are themselves the primary event.
- Avoid extracting multiple macro drivers that represent different aspects of the same underlying macroeconomic development.
- When multiple macro drivers are closely related, prefer fewer, broader drivers over many overlapping ones.
- Differentiate confirmed events from speculation. Assign lower confidence to hypothetical scenarios, anonymous sources or market speculation unless supported by official announcements or multiple independent reports.

Macro driver format:
- topic: Must be exactly one of: Federal Reserve, ECB, Interest Rates, Inflation, Employment, Economic Growth, Banking Sector, Government Debt, Trade, Energy, Oil, Geopolitics, Risk Sentiment, Financial Markets, Other.
- direction: Must align logically with the chosen topic:
  * For "Federal Reserve", "ECB", "Interest Rates", or "Inflation": Use "hawkish", "dovish", or "neutral". (Note: Rising inflation is hawkish, falling inflation is dovish).
  * For "Employment", "Economic Growth", "Banking Sector", "Government Debt", "Trade", "Risk Sentiment", or "Financial Markets": Use "improving", "worsening", or "neutral".
  * For "Geopolitics", "Energy", or "Oil": Use "escalating", "easing", or "neutral".
  * For "Other": Choose the most logical direction from the options above.
- affected_currency: Must be exactly one of: "usd", "eur", "both", "global", "other".
- status: Must be exactly one of: "new" (just announced), "update" (ongoing development), "resolved" (ended/concluded).
- strength: A value between 0.0 and 1.0 measuring how strongly the identified macroeconomic driver is supported by the supplied news, independent of its expected market impact.
- reason: One concise sentence explaining why this driver was identified.
Additional fields:
- risks: List the major macroeconomic risks or uncertainties mentioned in the news (MAX 3). Return an empty array if none are identified.
- importance: A value between 0.0 and 1.0 measuring the expected influence of the identified macroeconomic developments on EUR/USD over the next 1-3 candles. Use these anchors: (0.0 - 0.2 = Background noise; 0.3 - 0.5 = Confirmation or continuation of an active macro regime; 0.6 - 0.8 = Meaningful new data point or shift in narrative / Trigger; 0.9 - 1.0 = Massive systemic shock).
- confidence: A value between 0.0 and 1.0 reflecting how reliably the supplied news supports the extracted macroeconomic assessment, not the expected magnitude of future market movements.
Empty input handling:
If the supplied news is empty or contains no market-relevant macroeconomic information, return:
- an empty string for "summary",
- empty arrays for "macro_drivers" and "risks",
- importance = 0.0,
- confidence = 0.0.
Output ONLY valid JSON.
Do not include Markdown, code fences, explanations or additional text.
"""

system_prompt = """
You are a professional macroeconomic analyst specializing in foreign exchange markets.
You are given all relevant macroeconomic news published before the opening of the next 1-day EUR/USD candle.
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
                {"role": "system", "content": SYSTEM_PROMPT_NEW},
                {"role": "user", "content": f"News for candle {prediction_candle}:\n\n{news_data_str}"},
            ],
            response_format={"type": "json_object"},
        )
        
        llm_output = response.choices[0].message.content.strip()
        if llm_output.startswith("```"):
            llm_output = llm_output.strip("`").replace("json\n", "", 1).strip()
        
        try:
            parsed_analysis = json.loads(llm_output)
            print(f"{prediction_candle} processed")
        except json.JSONDecodeError:
            print(f"JSON-PARSE-ERROR at {prediction_candle}. Raw output saved.")
            parsed_analysis = {"raw_output": llm_output, "error": "JSON parse failed"}

        return {
            "prediction_candle": prediction_candle,
            "llm_analysis": parsed_analysis
        }

    except Exception as e:
        print(f"Error processing candle {prediction_candle}: {str(e)}")
        return {"prediction_candle": prediction_candle, "llm_analysis": {"error": str(e)}}

async def main():
    df = pd.read_csv(INPUT_CSV_PATH)
    df[CANDLE_COL] = pd.to_datetime(df[CANDLE_COL])
    df = df[(df[CANDLE_COL] >= START_DATE) & (df[CANDLE_COL] <= END_DATE)]
    df = df.sort_values(by=CANDLE_COL)
    df['V2Themes_Clean'] = df['V2Themes'].apply(lambda x: extract_top_themes(x, top_n=5))
    df['V2Organizations_Clean'] = df['V2Organizations'].apply(lambda x: extract_top_themes(x, top_n=3))
    df['V2Persons_Clean'] = df['V2Persons'].apply(lambda x: extract_top_themes(x, top_n=3))
    all_candles = pd.date_range(start=START_DATE, end=END_DATE, freq='12h')
    
    processed_candles = set()
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed_candles.add(data['prediction_candle'])
                    except json.JSONDecodeError:
                        continue
        print(f"Processed candles: {len(processed_candles)}. Skipped.")
    else:
        print("Export File not found. Start from beginning.")

    grouped = df.groupby(CANDLE_COL)
    tasks = []
    semaphore = asyncio.Semaphore(20)
    file_lock = asyncio.Lock()
    
    async def bound_process(candle_str, group):
        async with semaphore:
            result = await process_candle_news(candle_str, group)
        async with file_lock:
            with open(OUTPUT_JSON_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result
    
    empty_candles_written = 0
    for candle in all_candles:
        candle_str = candle.strftime("%Y-%m-%d %H:%M:%S")
        
        if candle_str in processed_candles:
            continue

        if candle in grouped.groups:
            group_df = grouped.get_group(candle)
            tasks.append(bound_process(candle_str, group_df))
        else:
            res = {"prediction_candle": candle_str, "llm_analysis": None}
            with open(OUTPUT_JSON_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            processed_candles.add(candle_str)
            empty_candles_written += 1

    if empty_candles_written > 0:
        print(f"Saved {empty_candles_written} empty candles instantly to file.")

    if tasks:
        print(f"Start processing the {len(tasks)} candles left")
        await asyncio.gather(*tasks)

    print("Sort candles in order.")
    
    final_results_dict = {}
    with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    res = json.loads(line)
                    final_results_dict[res['prediction_candle']] = res
                except json.JSONDecodeError:
                    continue
                
    all_results = list(final_results_dict.values())
    all_results.sort(key=lambda x: x['prediction_candle'])
    
    FINAL_JSON_PATH = "final_macro_signals_2020_2026_eet_eest_new_new_12h.json"
    with open(FINAL_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"Sorted saved in file: {FINAL_JSON_PATH}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
