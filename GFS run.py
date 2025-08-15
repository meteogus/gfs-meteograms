import re
import requests
from datetime import datetime, timedelta, timezone

# Get current UTC time
now_utc = datetime.now(timezone.utc)

# GFS run schedule
gfs_run_hours = [0, 6, 12, 18]

# Find the most recent run hour
latest_run_hour = max([h for h in gfs_run_hours if h <= now_utc.hour])
latest_run_time = now_utc.replace(hour=latest_run_hour, minute=0, second=0, microsecond=0)

# If before first run, go to yesterday's last run
if now_utc.hour < gfs_run_hours[0]:
    latest_run_time -= timedelta(days=1)

# Build NOAA directory URL for the latest run
url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{latest_run_time:%Y%m%d}/{latest_run_time:%H}/atmos/"

print(f"\n🌀 Latest GFS run: {latest_run_time:%Y-%m-%d %HZ}")

# Fetch directory listing
try:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
except requests.RequestException as e:
    print(f"❌ Could not access NOAA server: {e}")
    exit()

# Match forecast files
matches = re.findall(r"gfs\.t\d{2}z\.pgrb2\.0p25\.f(\d{3})", resp.text)

if matches:
    forecast_hours = [int(h) for h in matches]
    max_hour = max(forecast_hours)
    print(f"⏳ Maximum forecast hour available: {max_hour}h")
else:
    print("⚠️ No forecast files found yet for this run.")