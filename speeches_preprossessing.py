import asyncio
import pandas as pd
import json
import logging
from openai import AsyncOpenAI
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = AsyncOpenAI()
MODEL_NAME = "gemma4:31b"
INPUT_CSV_PATH = "all_ECB_speeches.csv"
OUTPUT_JSON_PATH = "processed_macro_signals_ecb_speeches_eet_eest_new_new.jsonl"
STATEMENTS_JSON_PATH = "final_macro_signals_ecb_statements_minutes.json"

system_prompt ="""
You are a highly specialized central bank analyst. 
You will be provided with a speech transcript from either the Federal Reserve (FED) or the European Central Bank (ECB), including the speaker's name and title.

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
- DISTINGUISH NEW VS. REITERATED: Your crucial task is to determine if the speaker is signaling a NEW shift in policy or simply REITERATING the already known stance from the last official meeting.
- If the stance is purely reiterated, set the "strength" and "importance" heavily towards 0.0. Only a clear deviation from previous guidance deserves high importance.
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
- importance: (0.0 to 1.0) Determine the market impact. Weight the speaker heavily. Speeches by the FED Chair or ECB President almost always warrant a higher importance than regional branch presidents. 
- confidence: (0.0 to 1.0) How clear and unambiguous is the speaker's language? 

Empty input handling:
If the text is empty or contains no relevant monetary policy signals, return:
- an empty string for "summary",
- empty arrays for "macro_drivers" and "risks",
- importance = 0.0,
- confidence = 0.0.

Output ONLY valid JSON. Do not include Markdown, code fences, explanations or additional text."""

system_prompt_new ="""
You are a highly specialized central bank analyst. 
You will be provided with a speech transcript from either the Federal Reserve (FED) or the European Central Bank (ECB), including the speaker's name and title.

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
- Determine whether the speech introduces a meaningful change in policy expectations, confirms the existing policy stance, or provides no meaningful policy guidance.
- Assign moderate importance when the speech mainly confirms the current policy stance. Assign high importance only if it materially changes policy expectations or clearly signals a deviation from the established stance.
- IGNORE any reference sections, bibliographies, or footnotes at the end of the transcript. Do NOT extract macro drivers from the titles of referenced academic papers.
- Evaluate the monetary policy tone. Focus heavily on Interest Rates, Forward Guidance, Inflation expectations, and Quantitative Tightening/Easing (QT/QE).
- If multiple monetary policy drivers are discussed, prioritize those most likely to influence future interest-rate expectations.
- Ignore discussion that is primarily academic, historical or institutional unless it materially affects monetary policy expectations.
- Repeated policy guidance should reinforce confidence in the existing stance, even if it does not substantially increase importance.
- IMPORTANT FOR FED: The Fed has a dual mandate (Inflation & Employment). Weigh labor market comments heavily.
- IMPORTANT FOR ECB: The ECB has a primary mandate (Inflation). Weigh inflation target comments heaviest.
- Summarize the overarching policy stance in MAX 3 sentences.
- List at most the 2 most market-relevant macro drivers.

Macro driver format:
- topic: Choose from: Forward Guidance, Interest Rates, Inflation, Employment/Labor, Economic Growth, QT/Balance Sheet, Financial Conditions.
- direction: Must be exactly one of: "hawkish" (restrictive, higher rates), "dovish" (accommodative, lower rates), "neutral".
- strength: A value between 0.0 and 1.0 indicating how strongly the identified macro driver is expressed within this speech, independent of its market impact.
- reason: One concise sentence quoting or explaining the driver.

Scoring rules:
- importance: (0.0 to 1.0) Importance measures the expected impact of this speech on monetary policy expectations over the next several trading sessions, not whether the information is completely new. Speeches by the Fed Chair, Fed Vice Chair, ECB President or ECB Vice President generally deserve higher importance than regional presidents, unless a regional president clearly signals a meaningful deviation from current policy expectations.
- confidence: (0.0 to 1.0) Confidence reflects how clearly the speaker's policy stance can be inferred from the speech, not whether the stance is hawkish or dovish.

Empty input handling:
If the text is empty or contains no relevant monetary policy signals, return:
- an empty string for "summary",
- empty arrays for "macro_drivers" and "risks",
- importance = 0.0,
- confidence = 0.0.

Output ONLY valid JSON. Do not include Markdown, code fences, explanations or additional text."""


# SYSTEM_PROMPT = """
# You are a highly specialized central bank analyst. 
# You will be provided with a speech transcript from either the Federal Reserve (FED) or the European Central Bank (ECB), including the speaker's name and title.

# Your ONLY task is to extract the monetary policy stance and forward guidance from the speech.
# Do NOT attempt to predict currency pairs. Focus purely on the central bank's inherent stance.

# Return ONLY valid JSON with the following structure:
# {
#   "summary": "...",
#   "macro_drivers": [
#     {
#       "topic": "...",
#       "direction": "...",
#       "strength": 0.0,
#       "reason": "..."
#     }
#   ],
#   "risks": [
#     "..."
#   ],
#   "importance": 0.0,
#   "confidence": 0.0
# }

# Guidelines:
# - IGNORE any reference sections, bibliographies, or footnotes at the end of the transcript. Do NOT extract macro drivers from the titles of referenced academic papers.
# - Evaluate the monetary policy tone. Focus heavily on Interest Rates, Forward Guidance, Inflation expectations, and Quantitative Tightening/Easing (QT/QE).
# - IMPORTANT FOR FED: The Fed has a dual mandate (Inflation & Employment). Weigh labor market comments heavily.
# - IMPORTANT FOR ECB: The ECB has a primary mandate (Inflation). Weigh inflation target comments heaviest.
# - Summarize the overarching policy stance in MAX 3 sentences.
# - List at most the 2 most important macro drivers.

# Macro driver format:
# - topic: Choose from: Forward Guidance, Interest Rates, Inflation, Employment/Labor, Economic Growth, QT/Balance Sheet, Financial Conditions.
# - direction: Must be exactly one of: "hawkish" (restrictive, higher rates), "dovish" (accommodative, lower rates), "neutral".
# - strength: A value between 0.0 and 1.0 indicating how explicitly this stance was communicated.
# - reason: One concise sentence quoting or explaining the driver.

# Scoring rules:
# - importance: (0.0 to 1.0) Determine the market impact. Weight the speaker heavily. Speeches by the FED Chair or ECB President almost always warrant a higher importance than regional branch presidents. 
# - confidence: (0.0 to 1.0) How clear and unambiguous is the speaker's language? 

# Empty input handling:
# If the text is empty or contains no relevant monetary policy signals, return:
# - an empty string for "summary",
# - empty arrays for "macro_drivers" and "risks",
# - importance = 0.0,
# - confidence = 0.0.

# Output ONLY valid JSON. Do not include Markdown, code fences, explanations or additional text.
# """
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

def load_json_with_dt(path, time_key, time_format):
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

async def process_window(index, speech, semaphore, file_lock, stats_list):
    async with semaphore:
        try:
            pure_date = datetime.strptime(speech["date"], "%Y-%m-%d").date()
            local_end_of_day = time(23, 59, 0)
            
            # us_tz = ZoneInfo("America/New_York")
            # localized_us_dt = datetime.combine(
            #     pure_date, local_end_of_day, tzinfo=us_tz
            # )

            # mt5_tz = ZoneInfo("Europe/Athens")
            # mt5_dt = localized_us_dt.astimezone(mt5_tz)
            # str_date = mt5_dt.strftime("%Y.%m.%d %H:%M")

            ecb_tz = ZoneInfo("Europe/Berlin")
            localized_ecb_dt = datetime.combine(
                pure_date, local_end_of_day, tzinfo=ecb_tz
            )

            mt5_tz = ZoneInfo("Europe/Athens")
            mt5_dt = localized_ecb_dt.astimezone(mt5_tz)
            str_date = mt5_dt.strftime("%Y.%m.%d %H:%M")
            speech_dt = pd.to_datetime(str_date, format="%Y.%m.%d %H:%M")

        except Exception as e:
            print(f"Error converting timezone for index {index} ({speech['date']}): {e}")
            return
        
        candidates = [x for x in stats_list if x['dt'] <= speech_dt]
        active_stats = []

        # 3. Aufbau des Referenzkontexts für den User-Prompt
        if candidates:
            # Finde das Datum des allerneuesten Events
            latest_event_dt = max(candidates, key=lambda x: x['dt'])['dt']
            latest_date = latest_event_dt.date()
            
            # Hole alle Events, die an exakt diesem Kalendertag stattfanden (z. B. Decision AND Q&A)
            active_stats = [x for x in candidates if x['dt'].date() == latest_date]
            # Sortiere sie chronologisch (damit die Decision vor dem Q&A steht)
            active_stats.sort(key=lambda x: x['dt'])

        # 3. Aufbau des dynamischen Referenzkontexts für das LLM
        if active_stats:
            reference_context = "### ACTIVE POLICY REFERENCE\n"
            reference_context += (
                "These are the official stances from the last meeting day. "
                "Evaluate the current speech against this baseline. Check if the speaker "
                "deviates from these official decisions and Q&A statements or just reiterates them:\n\n"
            )
            for idx, stat in enumerate(active_stats, 1):
                ref_headline = stat.get("headline", "Official Release")
                ref_summary = stat.get("summary", "No summary available.")
                ref_drivers = json.dumps(stat.get("macro_drivers", []), ensure_ascii=False)
                
                reference_context += (
                    f"Event {idx}: {ref_headline} (Released at {stat['time']})\n"
                    f"Summary: {ref_summary}\n"
                    f"Key Drivers: {ref_drivers}\n"
                    f"--------------------------------------------------\n"
                )
        else:
            reference_context = (
                "### ACTIVE POLICY REFERENCE\n"
                "No prior official central bank statement found in database. Analyze the speech purely on its own merits.\n"
            )

        clean_maintext = remove_references(speech['contents'])
        speech_str = (
            f"{reference_context}\n"
            f"=== CURRENT SPEECH TO ANALYZE ===\n"
            f"ECB speech from: {speech['speakers']}\n"
            f"Title: {speech['title']}\n"
            f"Subtitle: {speech['subtitle']}\n"
            f"Speech Date: {speech['date']}\n\n"
            f"Transcript:\n{clean_maintext}"
        )
        try:
            # Dispatch async request to the model
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt_new},
                    {"role": "user", "content": speech_str},
                ],
                response_format={"type": "json_object"},
            )
            
            raw_content = completion.choices[0].message.content
            if raw_content is None:
                print(f"Index {index}: API returned None (Empty response) at {str_date}")
                return
                            
            response = raw_content.strip()
            
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx+1]
            else:
                print(f"Index {index}: No JSON brackets found in response at time {str_date}.")
                return
                
            response_json = json.loads(response)
            response_json['time'] = str_date
            response_json["speaker"] = speech["speakers"]
            response_json["headline"] = speech["title"]
            async with file_lock:
                with open(OUTPUT_JSON_PATH, mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(response_json, ensure_ascii=False) + "\n")
                        
            print(f"Successfully processed speech {index} at time {str_date}")
            
        except json.JSONDecodeError:
            print(f"Error parsing JSON structure for window starting at  {index}.")
        except Exception as e:
            print(f"Exception encountered at window index {index}: {e}")

async def main():
    df = pd.read_csv(INPUT_CSV_PATH, sep='|')
    df = df.fillna("")
    df["date"] = df["date"].astype(str)
    relevant = df[df['date'].between('2020-01-01', '2026-06-02')].sort_values('date')

    print("Loading preprocessed ECB statements and minutes...")
    ecb_stats = load_json_with_dt(STATEMENTS_JSON_PATH, "time", "%Y.%m.%d %H:%M")
    print(f"Loaded {len(ecb_stats)} reference statements.")

    semaphore = asyncio.Semaphore(20)
    file_lock = asyncio.Lock()
    tasks = []
    # ecb fehler new: 105,262,288,380,449,500,584 (500 stayed none even after 5 retries, rest fixed after 1)
    # ecb fehler new_new: 449, 500 (none), 454 (no json brackets) (454 after 2 retries, 500and 449 returned none even after 10 retries, so manually added those with empty json)
    # fed fehler new: 
    # fed fehler new_new: 
    # for i in range(len(relevant)):
        
    for i in [449]:
        speech = relevant.iloc[i]
        # if speech['headline'] in processed_headlines:
        #     continue
        tasks.append(process_window(i, speech, semaphore, file_lock,ecb_stats))
        
    print(f"Dispatched {len(tasks)} async window tasks. Processing...")
    await asyncio.gather(*tasks)
    print("All tasks finished execution.")

    all_llm_responses = []
    with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    all_llm_responses.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                
    all_llm_responses.sort(key=lambda x: x['time'])
    
    FINAL_JSON_PATH = "final_macro_signals_ecb_speeches_eet_eest_new_new.json"
    with open(FINAL_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_llm_responses, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())