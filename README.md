This repository contains all documents, datasets and code used for the thesis.
First, follow the instructions of the GitHub from Fisichella and Garolla, as their developed Expert Advisor builds the baseline for this thesis.
https://github.com/marco82ger/AI_forex_fisichella_garolla

The used EA is the LSTM_AI_TI_system.mq5 and should be put in the /Experts/Advisors in the same directory of MT5.

The final data used in the MetaTrader 5 backtests consists of the following:

LLM signals:
extracted_signals_calendar_communication_gdelt_new_new_1h.csv

extracted_signals_calendar_communication_gdelt_new_new.csv

extracted_signals_calendar_communication_gdelt_new_new_12h.csv

extracted_signals_calendar_communication_gdelt_new_new_1d.csv

LSTM signals:

LSTM_predictions_H1_2020_2026.csv

LSTM_predictions_H4_2020_2026.csv

LSTM_predictions_H12_2020_2026.csv

LSTM_predictions_D1_2020_2026.csv

These need to be located into /Terminal/Common in the same diectory of MT5.

For GDELT preprocessing, gdelt_preprocessing_script.py inside the gdelt folder was used using gdelt_news_2020_2026_eet_eest.zip and news_data_multi_granularity.zip. The .zip files contain the .csv used.
For specches preprocessing, speeches_preprocessing.py was used with all_ECB_speeches.csv and fed_speeches_2020_2026 repectively.
For statements and minutes preprocessing, statements_minutes_preprocessing was utilized with fed_fomc_statements_and_minutes.csv and ecb_policies_and_minutes.csv.
The signals generation was done using the end_llm_h1.py, end_llm_h4.py, end_llm_h12.py, and end_llm_d1.py scripts.

All LSTM results were computed using LSTM.py.
