//+------------------------------------------------------------------+
//|                                                  BuyAndHold.mq5 |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade trade;

input double InpLotSize    = 0.1;
input ulong  InpMagicNum   = 999001; 

bool tradeExecuted = false;

int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNum);
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   if(!tradeExecuted && PositionsTotal() == 0)
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      
      if(trade.Buy(InpLotSize, _Symbol, ask, 0, 0, "Buy and Hold Baseline"))
      {
         Print("Successfully opened Buy position at: ", ask);
         tradeExecuted = true;
      }
      else
      {
         Print("Order execution failed. Error code: ", GetLastError());
      }
   }
}