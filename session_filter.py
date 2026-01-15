# ============================================================
# Session Filter – Kenya Time (UTC+3)
# Trades London & New York only
# ============================================================

from datetime import datetime, time

LONDON_START = time(11, 0)
LONDON_END   = time(19, 0)

NY_START = time(16, 0)
NY_END   = time(23, 59)

def session_allowed():
    now = datetime.now().time()
    return (LONDON_START <= now <= LONDON_END) or (NY_START <= now <= NY_END)
