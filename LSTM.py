import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)
tf.random.set_seed(seed_num)

df = pd.read_csv('EURUSD_H1_2010_2026.csv', parse_dates=['DateTime'], index_col='DateTime')
# df = pd.read_csv('EURUSD_H4_2010_2026.csv', parse_dates=['DateTime'], index_col='DateTime')
# df = pd.read_csv('EURUSD_H12_2010_2026.csv', parse_dates=['DateTime'], index_col='DateTime')
# df = pd.read_csv('EURUSD_D1_2000_2026.csv', parse_dates=['DateTime'], index_col='DateTime')

base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
df = df[base_features].copy()
horizon = 4
seq_length = 24

high_low_range = df['High'] - df['Low'] + 1e-8 
df['Body_Ratio'] = abs(df['Close'] - df['Open']) / high_low_range
df['Upper_Wick_Ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / high_low_range
df['Lower_Wick_Ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / high_low_range

simple_returns = df['Close'].pct_change()
df['Rolling_Vol_20'] = simple_returns.rolling(window=20).std()
df['Rolling_AutoCorr_20'] = simple_returns.rolling(window=20).apply(
    lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
)

df['Future_Close'] = df['Close'].shift(-horizon)
df.dropna(subset=['Future_Close'], inplace=True)
df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
df.drop(columns=['Future_Close'], inplace=True)

df.dropna(inplace=True)

for col in base_features:
    df[col] = np.log(df[col] / df[col].shift(1).replace(0, np.nan))

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

features = [
    'Open', 'High', 'Low', 
    'Close', 'Volume', 
    'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio',
    'Rolling_Vol_20', 'Rolling_AutoCorr_20'
]
def create_sequences(features_data, target_data,seq_length):
    X, y = [], []
    for i in range(len(features_data) - seq_length):
        X.append(features_data[i:(i + seq_length)])
        y.append(target_data[i + seq_length ])
    return np.array(X), np.array(y)


test_years = range(2020, 2027)
all_trade_decisions = []
all_validation_stats = [] 

train = 3
for test_year in test_years:
    print(f"\n--- Verarbeite Testjahr: {test_year} ---")
    
    train_start_str = f'{test_year - train}-01-01'
    # train_start_str = '2000-01-01'
    train_end_str = f'{test_year - 1}-12-31'
    
    test_start_str = f'{test_year}-01-01'
    test_end_str = f'{test_year}-12-31' if test_year < 2026 else '2026-06-02'
    
    df_train = df.loc[train_start_str:train_end_str].copy()
    
    if len(df_train) > horizon:
        df_train = df_train.iloc[:-horizon]
    
    buffer_start = pd.to_datetime(test_start_str) - pd.Timedelta(hours=24 * seq_length * 2)
    df_test_raw = df.loc[buffer_start:test_end_str].copy()
    
    if len(df_train) < seq_length or len(df_test_raw) <= seq_length:
        print(f"Überspringe {test_year} wegen unzureichender Daten.")
        continue

    scaler = RobustScaler() 
    
    train_features = df_train[features].values
    train_target = df_train['Target'].values
    
    test_features = df_test_raw[features].values
    test_target = df_test_raw['Target'].values
    
    scaled_train_X = scaler.fit_transform(train_features)
    scaled_test_X = scaler.transform(test_features)
    
    X_train_full, y_train_full = create_sequences(scaled_train_X, train_target, seq_length)
    
    split_idx = int(len(X_train_full) * 0.8)
    X_tr = X_train_full[:split_idx - horizon]
    y_tr = y_train_full[:split_idx - horizon]

    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    X_test, y_test = create_sequences(scaled_test_X, test_target, seq_length)
    
    if len(X_test) == 0:
        continue

    signal_dates_full = df_test_raw.index[seq_length - 1 : len(X_test) + seq_length - 1]
    execution_dates_full = df_test_raw.index[seq_length : len(X_test) + seq_length]   

    mask_test_year = (execution_dates_full >= test_start_str) & (execution_dates_full <= test_end_str)
    
    if not mask_test_year.any():
        continue
    
    X_test_year = X_test[mask_test_year]
    y_test_year = y_test[mask_test_year]
    execution_dates_year = execution_dates_full[mask_test_year]
    signal_dates_year = signal_dates_full[mask_test_year]
    
    inputs = Input(shape=(seq_length, len(features)))
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    lstm_out = LSTM(units=50, return_sequences=True)(x)
    attn_out = MultiHeadAttention(num_heads=2, key_dim=50)(
        query=lstm_out, key=lstm_out, value=lstm_out
    )
    attn_out = Dropout(0.2)(attn_out)
    pooled = GlobalAveragePooling1D()(attn_out)
    outputs = Dense(units=1, activation='sigmoid')(pooled)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_tr, y_tr,
        epochs=500,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0
    )

    # dynamic threshold search on validation data
    val_probs = model.predict(X_val, verbose=0).flatten()
    best_delta = 0.0
    best_acc = 0.0
    
    for delta in np.arange(0.0, 0.105, 0.005):
        lower = 0.5 - delta
        upper = 0.5 + delta
        mask = (val_probs <= lower) | (val_probs >= upper)
        coverage = np.sum(mask) / len(val_probs)
        
        if coverage > 0.01:
            filtered_probs = val_probs[mask]
            filtered_y = y_val[mask]
            preds = (filtered_probs > 0.5).astype(int)
            acc = np.mean(preds == filtered_y)
            
            if acc > best_acc:
                best_acc = acc
                best_delta = delta

    val_lower = 0.5 - best_delta
    val_upper = 0.5 + best_delta
    val_mask = (val_probs <= val_lower) | (val_probs >= val_upper)
    val_trades = np.sum(val_mask)
    
    val_correct = 0
    if val_trades > 0:
        val_filtered_probs = val_probs[val_mask]
        val_filtered_y = y_val[val_mask]
        val_preds = (val_filtered_probs > 0.5).astype(int)
        val_correct = np.sum(val_preds == val_filtered_y)
        
    all_validation_stats.append({
        'Test_Year': test_year,
        'Val_Total_Samples': len(val_probs),
        'Val_Trades': val_trades,
        'Val_Correct': val_correct
    })

    predictions_prob = model.predict(X_test_year, verbose=0).flatten()
    test_lower_bound = 0.5 - best_delta
    test_upper_bound = 0.5 + best_delta

    for i in range(len(predictions_prob)):
        prob_up = predictions_prob[i]
        actual_dir = y_test_year[i]
        
        if test_lower_bound < prob_up < test_upper_bound:
            signal = "HOLD"
            pred_dir = -1
            is_correct = None
        else:
            pred_dir = 1 if prob_up > 0.5 else 0
            signal = "BUY" if pred_dir == 1 else "SELL"
            is_correct = (pred_dir == actual_dir)
        
        all_trade_decisions.append({
            'Date': execution_dates_year[i],
            'Signal_Date': signal_dates_year[i],
            'Prob_Up': round(prob_up, 4),
            'Best_Delta': round(best_delta, 3),
            'Signal': signal,
            'Direction_Correct': is_correct,
            'Actual_Dir': actual_dir,
            'Pred_Dir': pred_dir
        })
        
results_df = pd.DataFrame(all_trade_decisions)
val_stats_df = pd.DataFrame(all_validation_stats)

if not results_df.empty:
    
    total_val_samples = val_stats_df['Val_Total_Samples'].sum()
    total_val_trades = val_stats_df['Val_Trades'].sum()
    total_val_correct = val_stats_df['Val_Correct'].sum()
    
    agg_val_acc = (total_val_correct / total_val_trades * 100) if total_val_trades > 0 else 0
    agg_val_coverage = (total_val_trades / total_val_samples * 100) if total_val_samples > 0 else 0
    
    print(f"\n==============================================")
    print(f"  AGGREGATED VALIDATION DATA SELECTION METRICS")
    print(f"  VALIDATION DA:           {agg_val_acc:.2f}%")
    print(f"  VAL MARKET COVERAGE:     {agg_val_coverage:.2f}% ({total_val_trades} Trades)")

    trades_df = results_df[results_df['Signal'] != "HOLD"]
    coverage_pct = (len(trades_df) / len(results_df)) * 100
    directional_accuracy = trades_df['Direction_Correct'].astype(int).sum() / len(trades_df) * 100
    
    print(f"  OOS ROLLING WINDOW DA: {directional_accuracy:.2f}%")
    print(f"  MARKTABDECKUNG (TRADES):              {coverage_pct:.2f}% ({len(trades_df)} Trades)")
    print(f"==============================================")
    results_df.to_csv("LSTM_predictions_H1_2020_2026.csv", index=False)


