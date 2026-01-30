# ============================================================
# ForexFactory News Filter for MT5 (Python)
# Full version with DEBUG output
# ============================================================

import requests
from datetime import datetime, timedelta, timezone
import time
import json
import os

# ================= CONFIG =================

NEWS_LOOKAHEAD_MIN = 30     # Minutes BEFORE news
NEWS_LOOKBACK_MIN  = 15     # Minutes AFTER news

BLOCK_IMPACT_LEVELS = {"High", "Medium"}

FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

REQUEST_TIMEOUT = 10  # seconds

CACHE_FILE = "forex_calendar_cache.json"
CACHE_EXPIRY_MIN = 60  # Cache valid for 60 minutes

# Request headers to appear as a browser
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


# ================= HELPERS =================

def extract_currencies(symbol: str):
    """
    Extract currencies from symbol
    EURUSD -> {'EUR', 'USD'}
    """
    symbol = symbol.upper()
    if len(symbol) < 6:
        return set()
    return {symbol[:3], symbol[3:6]}


def load_cached_events():
    """Load events from cache if still valid"""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        cache_time = datetime.fromisoformat(cache.get("timestamp", "1900-01-01"))
        age_min = (datetime.now(timezone.utc) - cache_time).total_seconds() / 60
        
        if age_min < CACHE_EXPIRY_MIN:
            return cache.get("events")
    except:
        pass
    
    return None


def save_cached_events(events):
    """Save events to cache"""
    try:
        cache = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "events": events
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass


# ================= MAIN LOGIC =================

def news_block_active(symbol="EURUSD", debug=False):
    """
    Returns True if Medium/High impact news is active
    """

    try:
        now = datetime.now(timezone.utc)
        currencies = extract_currencies(symbol)

        # Try to load from cache first
        events = load_cached_events()
        
        if events is None:
            # Not cached, fetch from API
            if debug:
                print("📡 Fetching fresh data from ForexFactory...")
            
            response = requests.get(
                FOREX_FACTORY_URL,
                timeout=REQUEST_TIMEOUT,
                headers=REQUEST_HEADERS
            )

            if response.status_code == 429:
                if debug:
                    print("⏳ Rate limited. Using cached data if available...")
                events = load_cached_events()
                if events is None:
                    if debug:
                        print("❌ Failed to fetch ForexFactory data (rate limited)")
                    return False
            elif response.status_code != 200:
                if debug:
                    print("❌ Failed to fetch ForexFactory data")
                return False
            else:
                events = response.json()
                save_cached_events(events)  # Cache successful response

        if debug:
            print("\nCurrent UTC Time:", now)
            print("Checking symbol:", symbol)
            print("Currencies:", currencies)
            print("-" * 65)

        for event in events:
            currency = event.get("currency")
            impact   = event.get("impact")
            date_str = event.get("date")

            if currency not in currencies:
                continue

            if impact not in BLOCK_IMPACT_LEVELS:
                continue

            event_time = datetime.fromisoformat(
                date_str.replace("Z", "+00:00")
            )

            delta_min = (event_time - now).total_seconds() / 60

            if debug:
                print(
                    f"{currency:3} | {impact:6} | "
                    f"{event_time} UTC | "
                    f"{delta_min:+.1f} min"
                )

            if -NEWS_LOOKBACK_MIN <= delta_min <= NEWS_LOOKAHEAD_MIN:
                if debug:
                    print("\n⛔ NEWS BLOCK ACTIVE ⛔")
                return True

        if debug:
            print("\n✅ No blocking news")

        return False

    except Exception as e:
        print("News filter error:", e)
        return False


# ================= TEST RUN =================

if __name__ == "__main__":
    symbol = "EURUSD"

    result = news_block_active(symbol, debug=True)

    print("\nFINAL RESULT:")
    print(f"News block active for {symbol}: {result}")
