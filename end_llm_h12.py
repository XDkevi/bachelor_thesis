import os
from openai import AsyncOpenAI
import pandas as pd
import json
import csv
import asyncio
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI()

SYSTEM_PROMPT_OPTIMIZED = """
You are a senior forex fundamental analyst specialized in EUR/USD macro trading.
Task: Analyze the provided chronological information and estimate the expected EUR/USD direction over the next 3 to 4 trading days.

The input may contain up to three information sources:
1. Economic calendar events
   - Structured macroeconomic releases with numerical values.
2. Official ECB/Fed communication
   - Speeches, statements, press conferences or meeting minutes providing monetary policy guidance.
3. Filtered macroeconomic news
   - News relevant to EUR/USD (e.g. geopolitics, energy, trade conflicts, financial stability).

Any section may be empty or omitted. Never assume information that is not provided.
Return exactly one valid JSON object. Do not output explanations, markdown, comments, code fences, or additional text before or after the JSON.

Decision Rules & Hierarchy
- Strict hold rule: If evidence is mixed, conflicting, or insufficient to form a strong conviction, you must default to "hold". Do not force a trade.
- REGIME OVER TRIGGER: A dominant monetary policy regime often persists across several trading sessions. Small or moderate economic calendar surprises rarely invalidate an established regime.
- If a short-term trigger conflicts with the dominant overarching trend (regime), default to "hold" unless the trigger is a massive, unexpected systemic shock. In a multi-day trading context, daily noise is easily absorbed by the macro regime.
- Treat the latest official central bank communication as the active policy guidance unless explicitly superseded.
- Multiple reports describing the same macro event should reinforce confidence rather than be treated as independent events.
- Lower your confidence score for vague, stale, or already priced-in information.

Economic Calendar Logic
- Focus strictly on the SURPRISE factor (Actual vs. Forecast). 
- Stronger-than-expected Euro Area data or Weaker-than-expected US data = EUR bullish.
- Weaker-than-expected Euro Area data or Stronger-than-expected US data = EUR bearish.
- Note: Calendar importance is provided on a 1-3 scale. Treat calendar importance as normalized to: 3 -> 1.0, 2 -> 0.66, 1 -> 0.33.

Output Format: Return exactly one valid JSON object. Create one item only for information that is actually present. Do not create placeholder or inferred items for empty sections:
{
  "items": [
    {
      "source": "calendar",
      "brief_rationale": "Short explanation of impact max 20 words",
      "relevance_score": 0.9,
      "confidence_score": 0.8,
      "direction": "EUR bearish"
    }
  ],
  "overall_view": {
    "regime_bias": "neutral",
    "trigger_bias": "bearish",
    "top_driver": {
        "source": "calendar",
        "headline": "MAX 8 words describing the single most important event"
    },
    "synthesis_rationale": "Explain how the pieces of evidence interact in max 2 sentences.",
    "overall_eurusd_bias": "sell",
    "primary_source": "calendar",
    "confidence_score": 0.85,
    "relevance_score": 0.9
  }
}

Field Definitions & Allowed Values:
- items[].source: "calendar", "communication", or "news"
- items[].direction: "EUR bullish", "EUR bearish", or "neutral"
- overall_view.regime_bias: "bullish", "bearish", or "neutral" (Underlying trend based on speeches and news)
- overall_view.trigger_bias: "bullish", "bearish", or "neutral" (Immediate impulse based on calendar data)
- overall_view.top_driver: An object containing "source" and "headline", or null if no drivers exist.
- overall_view.overall_eurusd_bias: "buy", "sell", or "hold" (The final traded direction. Must be "hold" if regime and trigger strongly conflict).
- overall_view.primary_source: "calendar", "communication", "news", or null
- overall_view.confidence_score: Float 0.0 to 1.0 (Confidence reflects the certainty of the directional assessment based on the data provided).
- overall_view.relevance_score: Float 0.0 to 1.0 (Relevance measures the expected influence on EUR/USD over the next 3 to 4 trading days).

Empty State Rule:
If all input sections (1, 2, and 3) are "None" or empty, the "items" array MUST be empty ([]), and you MUST return exactly this JSON object:
{
  "items": [],
  "overall_view": {
    "regime_bias": "neutral",
    "trigger_bias": "neutral",
    "top_driver": null,
    "synthesis_rationale": "No relevant information available.",
    "overall_eurusd_bias": "hold",
    "primary_source": null,
    "confidence_score": 0.0,
    "relevance_score": 0.0
  }
}
"""

RAW_RESPONSES_FILE = "all_raw_llm_responses_calendar_communication_gdelt_new_new_12h_test.jsonl"
EXTRACTED_METRICS_FILE = "extracted_signals_calendar_communication_gdelt_new_new_12h_test.csv"
def load_json_with_dt(path, time_key, time_format):
    """Hilfsfunktion zum Laden und Parsen von Datumsfeldern in JSONs."""
    if not os.path.exists(path):
        print(f"Warnung: Datei {path} nicht gefunden.")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        t_val = item.get(time_key)
        if t_val:
            if isinstance(t_val, str):
                t_val = t_val.replace("+00:00", "").replace("Z", "").strip()
                item[time_key] = t_val
            item['dt'] = pd.to_datetime(t_val, format=time_format)
        else:
            item['dt'] = pd.NaT
    return [x for x in data if pd.notna(x['dt'])]

async def process_candle(index, candle_time, semaphore, file_lock, combined_cal, ecb_speeches, fed_speeches, ecb_stats, fed_stats, gdelt_lookup):
    async with semaphore:
        news_data_str = f"=== CURRENT EVALUATION TIME: {candle_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n"
        window_start_48h = candle_time - pd.Timedelta(hours=48)
        cal_window = combined_cal[(combined_cal['time'] >= window_start_48h) & (combined_cal['time'] <= candle_time)]
        
        cal_window = cal_window.sort_values(by=['importance', 'time'], ascending=[False, False])
        top_10_cal = cal_window.head(10)
        
        news_data_str += "1. Economic calendar events (Last 48h):\n"
        if not top_10_cal.empty:
            for _, row in top_10_cal.iterrows():
                revised_val = row['revised_previous']
                revised_str = "N/A" if pd.isna(revised_val) or str(revised_val).lower() == 'nan' else revised_val
                news_data_str += (
                    f"Timestamp: {row['time'].strftime('%Y-%m-%d %H:%M:%S')} | Country: {row['country_code']} | "
                    f"Event: {row['event_name']} | Importance: {row['importance']} | "
                    f"Currency: {row['currency_code']} | Actual: {row['actual_value']} | "
                    f"Forecast: {row['forecast_value']} | Previous: {row['previous_value']} | "
                    f"Revised Prev: {revised_str}\n"
                )
        else:
            news_data_str += "None\n"
            

        news_data_str += "2. Official ECB/Fed communication:\n"
        
        fed_candidates = [x for x in fed_stats if x['dt'] <= candle_time]
        if fed_candidates:
            latest_fed_date = max(fed_candidates, key=lambda x: x['dt'])['dt'].date()
            latest_fed_events = sorted([x for x in fed_candidates if x['dt'].date() == latest_fed_date], key=lambda x: x['dt'])
            
            for ev in latest_fed_events:
                news_data_str += f"- Latest Fed Policy ({ev['time']}) | Headline: {ev['headline']} | Speaker: {ev['speaker']}\n  Summary: {ev['summary']}\n  Drivers: {json.dumps(ev['macro_drivers'])}\n"
        else:
            news_data_str += "- Latest Fed Policy: None\n"
            
        ecb_candidates = [x for x in ecb_stats if x['dt'] <= candle_time]
        if ecb_candidates:
            latest_ecb_date = max(ecb_candidates, key=lambda x: x['dt'])['dt'].date()
            latest_ecb_events = sorted([x for x in ecb_candidates if x['dt'].date() == latest_ecb_date], key=lambda x: x['dt'])
            
            for ev in latest_ecb_events:
                news_data_str += f"- Latest ECB ({ev['time']}) | Headline: {ev['headline']} | Speaker: {ev['speaker']}\n  Summary: {ev['summary']}\n  Drivers: {json.dumps(ev['macro_drivers'])}\n"
        else:
            news_data_str += "- Latest ECB Policy: None\n"
            
        window_start_120h = candle_time - pd.Timedelta(hours=120)
        recent_ecb_sp = [x for x in ecb_speeches if window_start_120h <= x['dt'] <= candle_time and x.get('summary') != ""]
        recent_fed_sp = [x for x in fed_speeches if window_start_120h <= x['dt'] <= candle_time and x.get('summary') != ""]
        all_recent_speeches = sorted(recent_ecb_sp + recent_fed_sp, key=lambda x: x['dt'], reverse=True)
        
        if all_recent_speeches:
            news_data_str += "- Recent Central Bank Speeches (Last 5 days):\n"
            for sp in all_recent_speeches[:8]:
                news_data_str += f"  * Timestamp: {sp['time']} | Speaker: {sp['speaker']} | Headline: {sp['headline']} | Importance: {sp['importance']}\n    Summary: {sp['summary']}\n    Drivers: {json.dumps(sp['macro_drivers'])}\n"
        else:
            news_data_str += "- Recent Central Bank Speeches: None\n"
            

        news_data_str += "3. Filtered macroeconomic news (Last 4 days):\n"
        news_timestamps = pd.date_range(end=candle_time, periods=8, freq='12h')
        
        for ts in news_timestamps:
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            item = gdelt_lookup.get(ts_str)
            if item and item.get('llm_analysis') and isinstance(item['llm_analysis'], dict) and item['llm_analysis'].get('summary'):
                analysis = item['llm_analysis']
                news_data_str += (
                    f"- News Window: {ts_str} | Importance: {analysis.get('importance')} | Confidence: {analysis.get('confidence')}\n"
                    f"  Summary: {analysis.get('summary')}\n"
                    f"  Drivers: {json.dumps(analysis.get('macro_drivers'))}\n"
                )
            else:
                news_data_str += f"- News Window: {ts_str} | No new news for this candle\n"
                
        try:
            completion = await client.chat.completions.create(
                model="gemma4:31b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_OPTIMIZED},
                    {"role": "user", "content": news_data_str},
                ],
            )
            
            raw_content = completion.choices[0].message.content.strip()
            
            if raw_content.startswith("```"):
                raw_content = raw_content.strip("`").replace("json\n", "", 1).strip()
                
            response_json = json.loads(raw_content)
            overall_view = response_json.get("overall_view", {})
            
            target_relevance = overall_view.get("relevance_score")
            target_confidence = overall_view.get("confidence_score")
            target_bias = overall_view.get("overall_eurusd_bias")
            
            async with file_lock:
                # Log JSONL
                with open(RAW_RESPONSES_FILE, mode='a', encoding='utf-8') as f:
                    log_wrapper = {"window_start_idx": index, "candle_timestamp": candle_time.strftime('%Y-%m-%d %H:%M:%S'), "raw_output": raw_content}
                    f.write(json.dumps(log_wrapper) + "\n")
                    
                # Log CSV
                with open(EXTRACTED_METRICS_FILE, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([index, candle_time.strftime('%Y-%m-%d %H:%M:%S'), target_relevance, target_confidence, target_bias])
                    
            print(f"Successfully processed candle {candle_time.strftime('%Y-%m-%d %H:%M:%S')} (Index {index})")
            
        except json.JSONDecodeError:
            print(f"Error parsing JSON structure for candle at {candle_time}.")
        except Exception as e:
            print(f"Exception encountered at candle {candle_time}: {e}")

async def main():
    print("Loading dataframes...")
    df_de_h = pd.read_csv("EUR_high_history_DE.csv", sep='\t')
    df_eu_h = pd.read_csv("EUR_high_history_EU.csv", sep='\t')
    df_us_h = pd.read_csv("USD_high_history.csv", sep='\t')
    df_de_m = pd.read_csv("EUR_medium_history_DE.csv", sep='\t')
    df_eu_m = pd.read_csv("EUR_medium_history_EU.csv", sep='\t')
    df_us_m = pd.read_csv("USD_medium_history.csv", sep='\t')
    df_es_m = pd.read_csv("EUR_medium_history_ES.csv", sep='\t')
    df_fr_m = pd.read_csv("EUR_medium_history_FR.csv", sep='\t')
    df_it_m = pd.read_csv("EUR_medium_history_IT.csv", sep='\t')
    
    all_cal_dfs = [df_de_h, df_eu_h, df_us_h, df_de_m, df_eu_m, df_us_m, df_es_m, df_fr_m, df_it_m]
    combined_cal = pd.concat(all_cal_dfs, ignore_index=True)
    combined_cal['time'] = pd.to_datetime(combined_cal['time'], format='%Y.%m.%d %H:%M:%S')
    
    print("Loading central bank communications & news...")
    ecb_speeches = load_json_with_dt("final_macro_signals_ecb_speeches_eet_eest_new_new.json", "time", "%Y.%m.%d %H:%M")
    fed_speeches = load_json_with_dt("final_macro_signals_fed_speeches_eet_eest_new_new.json", "time", "%Y.%m.%d %H:%M")
    ecb_stats = load_json_with_dt("final_macro_signals_ecb_statements_minutes.json", "time", "%Y.%m.%d %H:%M")
    fed_stats = load_json_with_dt("final_macro_signals_fed_statements_minutes.json", "time", "%Y.%m.%d %H:%M")
    gdelt_path = "gdelt/final_macro_signals_2020_2026_eet_eest_new_new_12h.json"
    gdelt_news = load_json_with_dt(gdelt_path, "prediction_candle", "%Y-%m-%d %H:%M:%S")
    
    gdelt_lookup = {item['prediction_candle']: item for item in gdelt_news}
    # test with no gdelt news
    # gdelt_lookup = {}

    if not os.path.exists(EXTRACTED_METRICS_FILE):
        with open(EXTRACTED_METRICS_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['window_start_idx','timestamp', 'relevance_score', 'confidence_score', 'overall_eurusd_bias'])

    GLOBAL_START = pd.Timestamp("2020-01-01 00:00:00")
    valid_dates_df = pd.read_csv("valid_h12_dates_2020_2026.csv")
    all_expected_candles = pd.to_datetime(valid_dates_df['DateTime'])
    # all_expected_candles = all_expected_candles[(all_expected_candles >= '2022-01-01') & (all_expected_candles < '2023-01-01')]
    processed_indices = set()
    if os.path.exists(EXTRACTED_METRICS_FILE):
        try:
            existing_df = pd.read_csv(EXTRACTED_METRICS_FILE, usecols=['window_start_idx'])
            processed_indices = set(existing_df['window_start_idx'].dropna().astype(int))
            print(f"Found {len(processed_indices)} already processed candles in CSV.")
        except Exception as e:
            print(f"Could not read existing metrics file or file is empty: {e}")

    candles_to_process = []
    for candle_time in all_expected_candles:
        global_index = int((candle_time - GLOBAL_START) / pd.Timedelta(hours=12))
        if global_index not in processed_indices:
            candles_to_process.append(candle_time)

    candles = pd.DatetimeIndex(candles_to_process)
    print(f"Remaining missing candles to process: {len(candles)}")
    if len(candles) > 0:
        MAX_CONCURRENT_REQUESTS = 30
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        file_lock = asyncio.Lock()

        tasks = []
        for candle_time in candles:
            global_index = int((candle_time - GLOBAL_START)/ pd.Timedelta(hours=12))
        
            tasks.append(
                process_candle(
                    global_index, candle_time, semaphore, file_lock, 
                    combined_cal, ecb_speeches, fed_speeches, 
                    ecb_stats, fed_stats, gdelt_lookup
                )
            )
        
        print(f"Dispatched {len(tasks)} async candle tasks. Processing...")
        await asyncio.gather(*tasks)
        print("All tasks finished execution.")
    else:
        print("All candles in the specified date range have already been processed!")

    if os.path.exists(EXTRACTED_METRICS_FILE):
        csv_df = pd.read_csv(EXTRACTED_METRICS_FILE)
        csv_df = csv_df.drop_duplicates(subset=['window_start_idx'], keep='last')
        csv_df = csv_df.sort_values(by='window_start_idx').reset_index(drop=True)
        csv_df.to_csv(EXTRACTED_METRICS_FILE, index=False)
        print(f"Successfully sorted {EXTRACTED_METRICS_FILE} chronologically.")

    if os.path.exists(RAW_RESPONSES_FILE):
        with open(RAW_RESPONSES_FILE, mode='r', encoding='utf-8') as f:
            jsonl_lines = [json.loads(line) for line in f]
            
        jsonl_lines.sort(key=lambda x: x.get('window_start_idx', 0))
        
        with open(RAW_RESPONSES_FILE, mode='w', encoding='utf-8') as f:
            for line in jsonl_lines:
                f.write(json.dumps(line) + "\n")
        print(f"Successfully sorted {RAW_RESPONSES_FILE} chronologically.")

if __name__ == "__main__":
    asyncio.run(main())
