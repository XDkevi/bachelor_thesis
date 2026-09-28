import json
import re
import pandas as pd
from pathlib import Path

# --- Configuration ---
LLM_FILES = {
    'H1': 'all_raw_llm_responses_calendar_communication_gdelt_new_new_1h.jsonl',
    'H4': 'all_raw_llm_responses_calendar_communication_gdelt_new_new.jsonl',
    'H12': 'all_raw_llm_responses_calendar_communication_gdelt_new_new_12h.jsonl',
    'D1': 'all_raw_llm_responses_calendar_communication_gdelt_new_new_1d.jsonl'
}

CSV_FILES = {
    'H1': 'EURUSD_H1_2010_2026.csv',
    'H4': 'EURUSD_H4_2010_2026.csv',
    'H12': 'EURUSD_H12_2010_2026.csv',
    'D1': 'EURUSD_D1_2000_2026.csv'
}

HORIZONS = {
    'H1': 4,   # 4 hours
    'H4': 6,   # 24 hours
    'H12': 7,  # 84 hours
    'D1': 4    # 96 hours
}

START_DATE = '2020-01-01'
END_DATE = '2026-06-01'


def process_csv_targets(csv_path, horizon):
    """Loads CSV, computes target signal using full history, and sets DateTime index."""
    df = pd.read_csv(csv_path, parse_dates=['DateTime'], index_col='DateTime')
    df.sort_index(inplace=True)
    
    # Remove index duplicates if any exist
    df = df[~df.index.duplicated(keep='first')]
    
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[c for c in base_features if c in df.columns]].copy()
    
    # Calculate target based on forward horizon (1 = Up/Buy, 0 = Down/Sell)
    df['Future_Close'] = df['Close'].shift(-horizon)
    df.dropna(subset=['Future_Close'], inplace=True)
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
    df.drop(columns=['Future_Close'], inplace=True)
    
    return df


def parse_jsonl_file(filepath):
    """Parses JSONL, extracts candle_timestamp, and maps strict schema bias outputs."""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                
                try:
                    outer_data = json.loads(line_str)
                    raw_output = outer_data.get('raw_output', '')
                    
                    if isinstance(raw_output, str):
                        raw_output = raw_output.strip()
                        inner_data = json.loads(raw_output)
                    elif isinstance(raw_output, dict):
                        inner_data = raw_output
                    else:
                        continue

                    if isinstance(inner_data, dict):
                        overall_view = inner_data.get('overall_view', {})
                        if isinstance(overall_view, dict) and overall_view:
                            dt = (outer_data.get('candle_timestamp'))
                            if dt:
                                overall_view['DateTime'] = dt
                            records.append(overall_view)

                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    
    expected_cols = [
        'overall_eurusd_bias', 'confidence_score', 'relevance_score', 
        'primary_source', 'regime_bias', 'trigger_bias'
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    bias_map = {'buy': 'Buy', 'sell': 'Sell', 'hold': 'Hold'}
    df['LLM_Signal'] = (
        df['overall_eurusd_bias']
        .astype(str)
        .str.strip()
        .str.lower()
        .map(bias_map)
        .fillna('Unknown')
    )
    
    return df


def analyze_aligned_data(llm_files, csv_files, horizons, start_date, end_date):
    all_merged_dfs = []

    for tf in llm_files.keys():
        llm_path = llm_files[tf]
        csv_path = csv_files[tf]
        horizon = horizons[tf]
        
        if not Path(llm_path).exists() or not Path(csv_path).exists():
            print(f"Missing file for {tf}.")
            continue
            
        df_csv = process_csv_targets(csv_path, horizon)
        df_llm = parse_jsonl_file(llm_path)
        
        if df_llm.empty or 'DateTime' not in df_llm.columns:
            print(f"Error {tf}: No valid DateTime extracted from JSONL.")
            continue

        df_llm['DateTime'] = pd.to_datetime(df_llm['DateTime'])
        df_llm.set_index('DateTime', inplace=True)
        df_llm = df_llm[~df_llm.index.duplicated(keep='first')]

        merged_df = df_csv.join(df_llm, how='inner')
        merged_df = merged_df.loc[start_date:end_date].copy()
        
        if merged_df.empty:
            print(f"Warning: No overlapping records found for {tf} between {start_date} and {end_date}.")
            continue

        merged_df['timeframe'] = tf
        all_merged_dfs.append(merged_df)

    if not all_merged_dfs:
        print("\nError: No valid data could be extracted across any timeframes.")
        return None

    combined_df = pd.concat(all_merged_dfs)

    print("\n=========================================")
    print("       AGGREGATED LLM STATISTICS         ")
    print("=========================================\n")
    print("Overall signal distribution (%)")
    signal_dist = combined_df.groupby('timeframe')['LLM_Signal'].value_counts(normalize=True).unstack(fill_value=0) * 100
    print(signal_dist.round(2).to_string(), "\n")

    combined_df['confidence_score'] = pd.to_numeric(combined_df['confidence_score'], errors='coerce')
    combined_df['relevance_score'] = pd.to_numeric(combined_df['relevance_score'], errors='coerce')
    
    print("Confidence and relevance scores (mean / std)")
    scores = combined_df.groupby('timeframe')[['confidence_score', 'relevance_score']].agg(['mean', 'std'])
    print(scores.round(3).to_string(), "\n")

    if 'primary_source' in combined_df.columns:
        print("Primary source (%)")
        source_dist = combined_df.groupby('timeframe')['primary_source'].value_counts(normalize=True).unstack(fill_value=0) * 100
        print(source_dist.round(2).to_string(), "\n")

    combined_df['bias_agreement'] = combined_df['regime_bias'] == combined_df['trigger_bias']
    agreement = combined_df.groupby('timeframe')['bias_agreement'].mean() * 100
    print("Regime/trigger agreement rate (%)")
    print(agreement.round(2).to_string(), "\n")

    print("=========================================")
    print("  YEARLY BIAS VS ACTUAL MARKET DIRECTION ")
    print("=========================================\n")
    
    combined_df['Year'] = combined_df.index.year
    
    for tf in combined_df['timeframe'].unique():
        print(f"--- TIMEFRAME: {tf} (Horizon: {HORIZONS[tf]} candles) ---")
        tf_df = combined_df[combined_df['timeframe'] == tf]
        
        yearly_stats = []
        for year, group in tf_df.groupby('Year'):
            total = len(group)
            if total == 0: 
                continue
                
            llm_sell_pct = (group['LLM_Signal'] == 'Sell').mean() * 100
            actual_down_pct = (group['Target'] == 0).mean() * 100  # Target 0 = Down
            
            yearly_stats.append({
                'Year': year,
                'N_Candles': total,
                'LLM_Sell_%': round(llm_sell_pct, 2),
                'Actual_Down_%': round(actual_down_pct, 2),
                'Bias_Gap_%': round(llm_sell_pct - actual_down_pct, 2)
            })
            
        yearly_df = pd.DataFrame(yearly_stats).set_index('Year')
        print(yearly_df.to_string())
        print("\n")

    return combined_df

if __name__ == "__main__":
    df_results = analyze_aligned_data(LLM_FILES, CSV_FILES, HORIZONS, START_DATE, END_DATE)