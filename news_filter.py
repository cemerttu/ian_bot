# ============================================================
# MT5 Economic Calendar News Filter
# Blocks EUR / USD medium & high impact news
# ============================================================

import MetaTrader5 as mt5
from datetime import datetime, timedelta

NEWS_LOOKAHEAD_MIN = 30   # minutes BEFORE news
NEWS_LOOKBACK_MIN  = 15   # minutes AFTER news

HIGH_IMPACT_LEVELS = {2, 3}   # 2 = Medium, 3 = High
RELEVANT_CURRENCIES = {"EUR", "USD"}

def news_block_active(symbol="EURUSD"):
    now = datetime.now()
    start = now - timedelta(minutes=NEWS_LOOKBACK_MIN)
    end   = now + timedelta(minutes=NEWS_LOOKAHEAD_MIN)

    try:
        # Try to get calendar events - method may not exist in older MT5 versions
        if hasattr(mt5, 'calendar_events'):
            events = mt5.calendar_events(start, end)
        else:
            # Fallback: calendar_events not available in this MT5 version
            return False
            
        if events is None:
            return False

        for e in events:
            if e.currency in RELEVANT_CURRENCIES and e.importance in HIGH_IMPACT_LEVELS:
                return True

        return False
    except (AttributeError, Exception):
        # If calendar events fail for any reason, don't block trading
        return False
