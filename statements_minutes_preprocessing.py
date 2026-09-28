import asyncio
import pandas as pd
import json
import logging
from openai import AsyncOpenAI
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import re
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = AsyncOpenAI()
MODEL_NAME = "gemma4:31b"
INPUT_CSV_PATH = "fed_fomc_statements_and_minutes.csv"
# INPUT_CSV_PATH = "ecb_policies_and_minutes.csv"
OUTPUT_JSON_PATH = "processed_macro_signals_fed_statements_minutes_eet_eest.jsonl"
# OUTPUT_JSON_PATH = "processed_macro_signals_ecb_statements_minutes_eet_eest.jsonl"

SYSTEM_PROMPT = """
You are a highly specialized central bank analyst. 
You will be provided with a monetary policy decision, monetary policy statement with Q&A, fomc statement or minutes from either the Federal Reserve (FED) or the European Central Bank (ECB), including the speaker's name and title.

Your ONLY task is to extract the monetary policy stance and forward guidance from the speech.
Do NOT attempt to predict currency pairs. Focus purely on the central bank's inherent stance.

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
- IGNORE any reference sections, bibliographies, or footnotes at the end of the transcript. Do NOT extract macro drivers from the titles of referenced academic papers.
- Evaluate the monetary policy tone. Focus heavily on Interest Rates, Forward Guidance, Inflation expectations, and Quantitative Tightening/Easing (QT/QE).
- IMPORTANT FOR FED: The Fed has a dual mandate (Inflation & Employment). Weigh labor market comments heavily.
- IMPORTANT FOR ECB: The ECB has a primary mandate (Inflation). Weigh inflation target comments heaviest.
- Summarize the overarching policy stance in MAX 3 sentences.
- List at most the 2 most important macro drivers.

Macro driver format:
- topic: Choose from: Forward Guidance, Interest Rates, Inflation, Employment/Labor, Economic Growth, QT/Balance Sheet, Financial Conditions.
- direction: Must be exactly one of: "hawkish" (restrictive, higher rates), "dovish" (accommodative, lower rates), "neutral".
- strength: A value between 0.0 and 1.0 indicating how explicitly this stance was communicated.
- reason: One concise sentence quoting or explaining the driver.

Scoring rules:
- importance: (0.0 to 1.0) Determine the market impact.
- confidence: (0.0 to 1.0) How confident are you in your decision?

Empty input handling:
If the text is empty or contains no relevant monetary policy signals, return:
- an empty string for "summary",
- empty arrays for "macro_drivers" and "risks",
- importance = 0.0,
- confidence = 0.0.

Output ONLY valid JSON. Do not include Markdown, code fences, explanations or additional text.
"""

def remove_references(text):
    if not isinstance(text, str):
        return ""
    
    split_keywords = [
        "\nReferences", 
        "\nREFERENCES", 
        "\nBibliography", 
        "\nFootnotes"
    ]
    
    for keyword in split_keywords:
        if keyword in text:
            text = text.split(keyword)[0]
    
    return text.strip()

async def process_window(index, document, semaphore, file_lock):
    async with semaphore:
        try:
            raw_time_str = str(document["time"]).strip()
            if len(raw_time_str.split(":")) == 2:
                raw_time_str += ":00"
                
            naive_dt = datetime.strptime(raw_time_str, "%Y.%m.%d %H:%M:%S")

            us_tz = ZoneInfo("America/New_York")
            ecb_tz = ZoneInfo("Europe/Berlin")
            localized_us_dt = naive_dt.replace(tzinfo=us_tz)
            localized_eu_dt = naive_dt.replace(tzinfo=ecb_tz)

            mt5_tz = ZoneInfo("Europe/Athens")
            mt5_dt = localized_us_dt.astimezone(mt5_tz)
            # mt5_dt= localized_eu_dt.astimezone(mt5_tz)
            str_date = mt5_dt.strftime("%Y.%m.%d %H:%M")

        except Exception as e:
            print(f"Error converting timezone for index {index} ({document['time']}): {e}")
            return

        clean_maintext = remove_references(document['maintext'])
        
        # European Central Bank or Federal Reserve
        doc_str = f"""Federal Reserve Official Document
                    Type/Headline: {document['type']}
                    Transcript: {clean_maintext}"""

        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": doc_str},
                ],
            )
            
            raw_content = completion.choices[0].message.content
            if raw_content is None:
                print(f"Index {index}: API returned None (Empty response) at {str_date}. Skipping.")
                return
                
            response = raw_content.strip()
            match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if match:
                clean_json_string = match.group(0)
            else:
                print(f"Index {index}: No JSON brackets found in response at {str_date}. Skipping.")
                return
                
            response_json = json.loads(clean_json_string)
            response_json['time'] = str_date
            response_json["speaker"] = "FOMC" # ECB Governing Council or FOMC
            response_json["headline"] = document["type"]
            
            async with file_lock:
                with open(OUTPUT_JSON_PATH, mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(response_json, ensure_ascii=False) + "\n")
                        
            print(f"Successfully processed document {index} at MT5-time {str_date}")
            
        except json.JSONDecodeError:
            print(f"Error parsing JSON structure for window starting at {index}.")
        except Exception as e:
            print(f"Exception encountered at window index {index}: {e}")

def load_robust_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    pattern = re.compile(r'(?m)^(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\|')
    
    parts = pattern.split(content)
    
    data = []
    for i in range(1, len(parts), 2):
        time_str = parts[i]
        rest_of_row = parts[i+1]
        
        row_parts = rest_of_row.split('|')
        if len(row_parts) >= 3:
            doc_type = row_parts[0]
            url = row_parts[-1].strip()
            maintext = "|".join(row_parts[1:-1]).strip()
            
            if maintext.startswith('"') and maintext.endswith('"'):
                maintext = maintext[1:-1]
                
            data.append({
                "time": time_str,
                "type": doc_type,
                "maintext": maintext,
                "url": url
            })
            
    return pd.DataFrame(data)

async def main():
    df = load_robust_csv(INPUT_CSV_PATH)
    df = df.fillna("")
    
    df["time"] = df["time"].astype(str)
    relevant = df.sort_values('time')

    semaphore = asyncio.Semaphore(20)
    file_lock = asyncio.Lock()
    total_rows = len(relevant)
    tasks = []
    # fed fehler indices: 21,37,3
    # ecb fehler indices: 148
    # for i in[37]:
    for i in range(0, total_rows):
        document = relevant.iloc[i]
        tasks.append(process_window(i, document, semaphore, file_lock))
        
    print(f"Dispatched {len(tasks)} async window tasks. Processing...")
    await asyncio.gather(*tasks)
    print("All tasks finished execution.")

    all_llm_responses = []
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        all_llm_responses.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                
    all_llm_responses.sort(key=lambda x: x['time'])
    
    FINAL_JSON_PATH = "final_macro_signals_fed_statements_minutes.json"
    # FINAL_JSON_PATH = "final_macro_signals_ecb_statements_minutes.json"
    with open(FINAL_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_llm_responses, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())