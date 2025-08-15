import requests
from datetime import datetime, timezone, timedelta

# ---------- Step 1: Latest GFS run ----------
# GFS runs at 00, 06, 12, 18 UTC
gfs_run_hours = [0, 6, 12, 18]
now_utc = datetime.now(timezone.utc)
latest_run_hour = max([h for h in gfs_run_hours if h <= now_utc.hour])
latest_gfs_run = now_utc.replace(hour=latest_run_hour, minute=0, second=0, microsecond=0)

# If current time before first run, use yesterday's last run
if now_utc.hour < gfs_run_hours[0]:
    latest_gfs_run -= timedelta(days=1)

# NOAA URL to list forecast files
noaa_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{latest_gfs_run:%Y%m%d}/{latest_gfs_run:%H}/atmos/"

# ---------- Step 2: Get NOAA max forecast hour ----------
try:
    resp = requests.get(noaa_url, timeout=10)
    resp.raise_for_status()
    import re
    matches = re.findall(r"gfs\.t\d{2}z\.pgrb2\.0p25\.f(\d{3})", resp.text)
    if matches:
        noaa_max_hour = max(int(h) for h in matches)
    else:
        noaa_max_hour = None
except Exception as e:
    print(f"❌ Error accessing NOAA: {e}")
    noaa_max_hour = None

# ---------- Step 3: Get Open-Meteo latest forecast ----------
# Example: latitude & longitude (you can change)
lat, lon = 38.0, 23.7  # Athens
open_meteo_url = f"https://api.open-meteo.com/v1/gfs?latitude={lat}&longitude={lon}&hourly=temperature_2m"

try:
    resp = requests.get(open_meteo_url, timeout=10).json()
    times = resp['hourly']['time']
    last_time_str = times[-1]  # latest forecast time
    last_forecast_time = datetime.fromisoformat(last_time_str).replace(tzinfo=timezone.utc)
    open_meteo_max_hour = int((last_forecast_time - latest_gfs_run).total_seconds() / 3600)
except Exception as e:
    print(f"❌ Error accessing Open-Meteo: {e}")
    open_meteo_max_hour = None

# ---------- Step 4: Display ----------
print(f"\n🌀 Latest GFS run: {latest_gfs_run:%Y-%m-%d %HZ}")
if noaa_max_hour is not None:
    print(f"🌐 NOAA max forecast hour: f{noaa_max_hour}")
else:
    print("⚠️ Could not get NOAA max forecast hour")

if open_meteo_max_hour is not None:
    print(f"⏱ Open-Meteo max forecast hour available: {open_meteo_max_hour}h")
else:
    print("⚠️ Could not get Open-Meteo forecast hours")

if noaa_max_hour and open_meteo_max_hour:
    delay = noaa_max_hour - open_meteo_max_hour
    print(f"⏳ Open-Meteo is behind NOAA by ~{delay} forecast hours")