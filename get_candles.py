import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import timezonefinder, pytz

if not mt5.initialize():
    print("MetaTrader5 Initialisierung fehlgeschlagen. Fehler-Code:", mt5.last_error())
    quit()

symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H4  

start_date = datetime(2000, 1, 1, tzinfo=pytz.utc)
end_date = datetime(2026, 6, 2, tzinfo=pytz.utc)

print(f"Lade Daten für {symbol} von {start_date.strftime('%Y-%m-%d')} bis {end_date.strftime('%Y-%m-%d')}...")

rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
mt5.shutdown()

if rates is not None and len(rates) > 0:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    filename = f"{symbol}_H4_2000_2026.csv"
    df.to_csv(filename, index=False)
    print(f"Erfolgreich! {len(df)} Kerzen wurden in '{filename}' gespeichert.")
else:
    print("Keine Daten erhalten. Überprüfe das Symbol oder die Terminal-Verbindung.")