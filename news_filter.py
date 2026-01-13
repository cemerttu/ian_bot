import requests
from datetime import datetime, timedelta

NEWS_LOOKAHEAD_MIN = 30
NEWS_LOOKBACK_MIN = 15
HIGH_IMPACT_LEVELS = {"High", "Medium"}  # depends on API
RELEVANT_CURRENCIES = {"EUR", "USD"}

def news_block_active(symbol="EURUSD"):
    now = datetime.utcnow()
    start = now - timedelta(minutes=NEWS_LOOKBACK_MIN)
    end = now + timedelta(minutes=NEWS_LOOKAHEAD_MIN)

    # Example API call (Trading Economics, free plan)
    url = "https://api.tradingeconomics.com/calendar/country/united%20states?c=guest:guest&f=json"
    resp = requests.get(url)
    events = resp.json() if resp.status_code == 200 else []

    for ev in events:
        try:
            currency = ev.get("Country", "").upper()
            impact = ev.get("Impact", "")
            date_str = ev.get("Date", "")
            event_time = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
            
            if currency not in RELEVANT_CURRENCIES:
                continue
            if impact not in HIGH_IMPACT_LEVELS:
                continue

            diff_min = abs((event_time - now).total_seconds()) / 60
            if diff_min <= NEWS_LOOKAHEAD_MIN:
                reason = f"{currency} NEWS | Impact {impact} | {event_time.strftime('%H:%M')}"
                return True, reason
        except Exception:
            continue

    return False, ""

if __name__ == "__main__":
    blocked, reason = news_block_active("EURUSD")
    if blocked:
        print("🚫 NEWS BLOCK:", reason)
    else:
        print("✅ NO NEWS — TRADING ALLOWED")
