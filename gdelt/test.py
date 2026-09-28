import pandas as pd

def add_timeframe_candles(
    file_path: str, 
    timeframes_hours: list[int] = [1, 12, 24], 
    tz: str = "Europe/Athens"
) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    # 1. UTC-Timestamp parsen und in EET/EEST umwandeln
    ts_utc = pd.to_datetime(df['PublishTimestamp'], utc=True)
    ts_local = ts_utc.dt.tz_convert(tz).dt.tz_localize(None)
    
    # 2. Kerzen-Grenzen für jede Granularität berechnen
    for hours in timeframes_hours:
        suffix = "1D" if hours == 24 else f"{hours}H"
        
        # Auf volle Stunde abrunden und Offset zum Intervallstart berechnen
        ts_floored_hour = ts_local.dt.floor('h')
        hour_offset = pd.to_timedelta(ts_local.dt.hour % hours, unit='h')
        
        # Candle- und PredictionCandle-Startzeiten
        candle_start = ts_floored_hour - hour_offset
        prediction_candle = candle_start + pd.Timedelta(hours=hours)
        
        # Neue Spalten im Datensatz anlegen
        df[f'Candle_{suffix}'] = candle_start.dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df[f'PredictionCandle_{suffix}'] = prediction_candle.dt.strftime('%Y-%m-%d %H:%M:%S%z')
        
    return df

# Datensatz verarbeiten und speichern
df_multi_granularity = add_timeframe_candles("gdelt_news_2020_2026_eet_eest.csv")
df_multi_granularity.to_csv("news_data_multi_granularity.csv", index=False)