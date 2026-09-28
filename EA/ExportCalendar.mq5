//+------------------------------------------------------------------+
//|                                               ExportCalendar.mq5 |
//+------------------------------------------------------------------+
#property script_show_inputs

input datetime FROM_DATE = D'2020.01.01 00:00';
input datetime TO_DATE   = D'2026.06.01 23:59';

string NormalizeDateToServerCutoff(datetime requested_to)
{
   datetime server_now = TimeTradeServer();
   if(requested_to == 0 || requested_to > server_now)
      return TimeToString(server_now, TIME_DATE|TIME_SECONDS);
   return TimeToString(requested_to, TIME_DATE|TIME_SECONDS);
}

//+------------------------------------------------------------------+
//| Export function                                                  |
//+------------------------------------------------------------------+
void ExportHighImportanceHistory(string country_code, string currency_code, string filename)
{
   datetime server_now = TimeTradeServer();
   datetime to_date = TO_DATE;
   if(to_date == 0 || to_date > server_now)
      to_date = server_now;

   MqlCalendarValue values[];
   int count = CalendarValueHistory(values, FROM_DATE, to_date, country_code, currency_code);

   if(count <= 0)
   {
      Print("No calendar values found for ", country_code, "/", currency_code,
            " | Error=", GetLastError());
      return;
   }

   Print("Found ", count, " raw calendar values for ", country_code, "/", currency_code);

   int file_handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(file_handle == INVALID_HANDLE)
   {
      Print("Failed to open file: ", filename, " | Error=", GetLastError());
      return;
   }

   FileWrite(file_handle,
             "event_id",
             "event_name",
             "time",
             "actual_value",
             "forecast_value",
             "previous_value",
             "revised_previous",
             "importance",
             "country_code",
             "currency_code");

   int exported_rows = 0;

   for(int i = 0; i < count; i++)
   {
      MqlCalendarValue val = values[i];

      if(val.time > server_now)
         continue;

      if(!val.HasActualValue())
         continue;

      MqlCalendarEvent event_data;
      if(!CalendarEventById(val.event_id, event_data))
         continue;

      // Nur hohe Relevanz
      if(event_data.importance != CALENDAR_IMPORTANCE_HIGH)
         continue;

      FileWrite(file_handle,
                val.event_id,
                event_data.name,
                TimeToString(val.time, TIME_DATE | TIME_SECONDS),
                val.GetActualValue(),
                val.GetForecastValue(),
                val.GetPreviousValue(),
                val.GetRevisedValue(),
                event_data.importance,
                country_code,
                currency_code);

      exported_rows++;
   }

   FileClose(file_handle);

   Print("Export finished: ", filename, " | rows exported: ", exported_rows);
}
void ExportMediumImportanceHistory(string country_code, string currency_code, string filename)
{
   datetime server_now = TimeTradeServer();
   datetime to_date = TO_DATE;
   if(to_date == 0 || to_date > server_now)
      to_date = server_now;

   MqlCalendarValue values[];
   int count = CalendarValueHistory(values, FROM_DATE, to_date, country_code, currency_code);

   if(count <= 0)
   {
      Print("No calendar values found for ", country_code, "/", currency_code,
            " | Error=", GetLastError());
      return;
   }

   Print("Found ", count, " raw calendar values for ", country_code, "/", currency_code);

   int file_handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(file_handle == INVALID_HANDLE)
   {
      Print("Failed to open file: ", filename, " | Error=", GetLastError());
      return;
   }

   FileWrite(file_handle,
             "event_id",
             "event_name",
             "time",
             "actual_value",
             "forecast_value",
             "previous_value",
             "revised_previous",
             "importance",
             "country_code",
             "currency_code");

   int exported_rows = 0;

   for(int i = 0; i < count; i++)
   {
      MqlCalendarValue val = values[i];

      if(val.time > server_now)
         continue;

      if(!val.HasActualValue())
         continue;

      MqlCalendarEvent event_data;
      if(!CalendarEventById(val.event_id, event_data))
         continue;

      // Nur mittlere Relevanz
      if(event_data.importance != CALENDAR_IMPORTANCE_MODERATE)
         continue;

      FileWrite(file_handle,
                val.event_id,
                event_data.name,
                TimeToString(val.time, TIME_DATE | TIME_SECONDS),
                val.GetActualValue(),
                val.GetForecastValue(),
                val.GetPreviousValue(),
                val.GetRevisedValue(),
                event_data.importance,
                country_code,
                currency_code);

      exported_rows++;
   }

   FileClose(file_handle);

   Print("Export finished: ", filename, " | rows exported: ", exported_rows);
}
//+------------------------------------------------------------------+
//| Script start                                                     |
//+------------------------------------------------------------------+
void OnStart()
{
   if(MQLInfoInteger(MQL_TESTER))
   {
      Print("Calendar functions are not supported in the Strategy Tester.");
      return;
   }

   ExportHighImportanceHistory("US", "USD", "USD_high_history.csv");
   ExportHighImportanceHistory("EU", "EUR", "EUR_high_history_EU.csv");
   ExportHighImportanceHistory("DE", "EUR", "EUR_high_history_DE.csv");
   ExportHighImportanceHistory("FR", "EUR", "EUR_high_history_FR.csv");
   ExportHighImportanceHistory("IT", "EUR", "EUR_high_history_IT.csv");
   ExportHighImportanceHistory("ES", "EUR", "EUR_high_history_ES.csv");
   ExportMediumImportanceHistory("US", "USD", "USD_medium_history.csv");
   ExportMediumImportanceHistory("EU", "EUR", "EUR_medium_history_EU.csv");
   ExportMediumImportanceHistory("DE", "EUR", "EUR_medium_history_DE.csv");
   ExportMediumImportanceHistory("FR", "EUR", "EUR_medium_history_FR.csv");
   ExportMediumImportanceHistory("IT", "EUR", "EUR_medium_history_IT.csv");
   ExportMediumImportanceHistory("ES", "EUR", "EUR_medium_history_ES.csv");

   Print("All exports completed.");
}