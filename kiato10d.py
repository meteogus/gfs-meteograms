import numpy as np
import pandas as pd
import requests
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

from datetime import datetime, timedelta, timezone
from collections import defaultdict

# Get current UTC time
now_utc = datetime.now(timezone.utc)

# GFS runs at 00Z, 06Z, 12Z, and 18Z
gfs_run_hours = [0, 6, 12, 18]

# Find the most recent GFS run
latest_run_hour = max([h for h in gfs_run_hours if h <= now_utc.hour])
latest_run_time = now_utc.replace(hour=latest_run_hour, minute=0, second=0, microsecond=0)

# If current time is earlier than first run (00Z), go back one day
if now_utc.hour < gfs_run_hours[0]:
    latest_run_time -= timedelta(days=1)

print(f"Latest GFS run: {latest_run_time:%Y-%m-%d %HZ}")

# Location
latitude = 38.01
longitude = 22.75

# API call parameters
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": ",".join([
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "pressure_msl",
        "windspeed_10m",
        "wind_gusts_10m", 
        "winddirection_10m",
        "precipitation",            
        "showers",         
        "snowfall",
        "temperature_2m",
    ]),
    "forecast_days": 11,
    "timezone": "UTC",
    "models": "gfs_global"
}

max_total_wait = 900
delay_seconds = 60
timeout_per_request = 30
max_retries = max_total_wait // delay_seconds

for attempt in range(1, max_retries + 1):
    try:
        response = requests.get(url, params=params, timeout=timeout_per_request)
        response.raise_for_status()
        data = response.json()
        break
    except (requests.exceptions.RequestException) as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt == max_retries:
            print("Max retries reached. Exiting.")
            raise
        else:
            print(f"Retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)

times_cloud = pd.to_datetime(data['hourly']['time']).tz_localize('UTC')
cloud_low = np.array(data['hourly']['cloud_cover_low'])
cloud_mid = np.array(data['hourly']['cloud_cover_mid'])
cloud_high = np.array(data['hourly']['cloud_cover_high'])

def get_data(name):
    return np.array(data['hourly'][name])

pressure_msl = get_data('pressure_msl')
windspeed_10m = get_data('windspeed_10m')
winddirection_10m = get_data('winddirection_10m')
precipitation = get_data('precipitation')
showers = get_data('showers')
snowfall = get_data('snowfall')
wind_gusts_10m = get_data("wind_gusts_10m")
temperature_2m = get_data('temperature_2m')


# --- FILTER DATA FROM GFS RUN TIME ONWARD ---
start_time = latest_run_time
start_index = np.where(times_cloud >= start_time)[0][0]
end_index = start_index + 264
if end_index >= len(times_cloud):
    end_index = len(times_cloud) - 1

# Slice times and recalc numeric times for matplotlib
times = times_cloud[start_index:end_index + 1]
time_nums = mdates.date2num(times)

# Slice cloud layers
cloud_low = cloud_low[start_index:end_index + 1]
cloud_mid = cloud_mid[start_index:end_index + 1]
cloud_high = cloud_high[start_index:end_index + 1]

# Slice surface and other variables
pressure_msl = pressure_msl[start_index:end_index + 1]
windspeed_10m = windspeed_10m[start_index:end_index + 1]
winddirection_10m = winddirection_10m[start_index:end_index + 1]
precipitation = precipitation[start_index:end_index + 1]
showers = showers[start_index:end_index + 1]
snowfall = snowfall[start_index:end_index + 1]
wind_gusts_10m = wind_gusts_10m[start_index:end_index + 1]
temperature_2m = temperature_2m[start_index:end_index + 1]

print(f"Data filtered from {times[0]:%Y-%m-%d %HZ} to {times[-1]:%Y-%m-%d %HZ}")

# --- Plotting ---
fig, axs = plt.subplots(
    5, 1,
    figsize=(1000 / 96, 600 / 96), 
    gridspec_kw={'height_ratios': [0.8, 1, 0.8, 1, 1]},
    sharex=True
)






# --- Section 1: Clouds ---
ax_cloud = axs[0]
ax_cloud.set_facecolor('#00A0FF')
ax_cloud.set_ylim(0, 3)
ax_cloud.set_yticks([0.5, 1.5, 2.5])
ax_cloud.set_yticklabels(['Low', 'Mid', 'High'], fontsize=10, color='black')
time_nums = mdates.date2num(times)
dt_cloud = time_nums[1] - time_nums[0]

for cloud_cover, band_center in zip([cloud_low, cloud_mid, cloud_high], [0.5, 1.5, 2.5]):
    for i in range(len(times)):
        height = 0.6 * (cloud_cover[i] / 100)
        if height > 0:
            ax_cloud.axhspan(
                band_center - height / 2,
                band_center + height / 2,
                xmin=(time_nums[i] - dt_cloud / 2 - time_nums[0]) / (time_nums[-1] - time_nums[0]),
                xmax=(time_nums[i] + dt_cloud / 2 - time_nums[0]) / (time_nums[-1] - time_nums[0]),
                color='white',
                alpha=1.0
            )

ax_cloud.set_title(
    f"KIATO Init: {latest_run_time:%Y-%m-%d} ({latest_run_time:%HZ})",
    loc="center", fontsize=12, fontweight='bold', color='black', y=1.4
)
ax_cloud.tick_params(axis='y', colors='black')
ax_cloud.set_xlim(time_nums[0], time_nums[-1])
ax_cloud.grid(axis='x', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)






# --- Section 2: Precipitation ---
ax_precip = axs[1]
bar_width = (time_nums[1] - time_nums[0]) * 1.8
bar_width_showers = (time_nums[1] - time_nums[0]) * 0.9

rain = np.array(precipitation)          
showers_arr = np.array(showers)         
snowfall_arr = np.array(snowfall)       
time_arr = np.array(time_nums)

n = (len(rain) // 3) * 3
rain_3h = rain[:n].reshape(-1, 3).sum(axis=1)
showers_3h = showers_arr[:n].reshape(-1, 3).sum(axis=1)
snowfall_3h = snowfall_arr[:n].reshape(-1, 3).sum(axis=1)
time_nums_3h = time_arr[:n].reshape(-1, 3)[:, 0]

ax_precip.bar(time_nums_3h, rain_3h, width=bar_width, color='#20D020', alpha=1.0)
ax_precip.bar(time_nums_3h, showers_3h, width=bar_width_showers, color='#FA3C3C', alpha=1.0)
ax_precip.bar(time_nums_3h, snowfall_3h, width=bar_width, color='#4040FF', alpha=1.0)

ax_precip.set_ylabel('Precip.\n(mm)', fontsize=10, color='black')
ax_precip.tick_params(axis='y', labelcolor='black')
ax_precip.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# --- Adjust y-axis scale (start at 0, but no 0 tick label) ---
max_precip = max(np.max(rain_3h), np.max(showers_3h), np.max(snowfall_3h))
y_step = 2 if max_precip <= 10 else 5 if max_precip <= 30 else 10
y_max = y_step * np.ceil((max_precip + 2) / y_step)

ax_precip.set_ylim(0, y_max)
ax_precip.set_yticks(np.arange(y_step, y_max + y_step, y_step))

# Fix right border:








# --- Section 3: Pressure ---
ax_pressure = axs[2]
ax_pressure.plot(times, pressure_msl, color='#00A0FF', linewidth=1, label='SLP (hPa)')
ax_pressure.set_ylabel('SLP\n(hPa)', fontsize=10, color='black')
ax_pressure.tick_params(axis='y', labelcolor='black')
ax_pressure.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# Rounded min/max and step
pmin_rounded = int(np.floor(pressure_msl.min() / 5.0) * 5)
pmax_rounded = int(np.ceil(pressure_msl.max() / 5.0) * 5)
p_range = pmax_rounded - pmin_rounded
step = 10 if p_range > 20 else 5
ax_pressure.set_ylim(pmin_rounded, pmax_rounded)

# Set ticks
yticks = np.arange(pmin_rounded, pmax_rounded + 1, step)
ax_pressure.set_yticks(yticks)








# --- Section 4: Temperature 2m ---
ax_temp = axs[3]  # your 4th subplot
ax_temp.plot(times, temperature_2m, color='#FF8000', linewidth=1.5)
ax_temp.set_ylabel('T2m\n(°C)', fontsize=10, color='black')
ax_temp.tick_params(axis='y', labelcolor='black')
ax_temp.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# --- Y-axis limits and ticks ---
tmin_rounded = int(np.floor(temperature_2m.min() / 2.0) * 2)
tmax_actual = temperature_2m.max()
tmax_with_margin = int(np.ceil(tmax_actual)) + 4   # add +4 margin for annotation boxes

ax_temp.set_ylim(tmin_rounded, tmax_with_margin)

yticks = np.arange(tmin_rounded, tmax_with_margin + 1, 2)
ax_temp.set_yticks(yticks)

# Hide the top tick label
yticklabels = [str(t) for t in yticks]
yticklabels[-1] = ''
ax_temp.set_yticklabels(yticklabels)

# --- Plot daily maxima between 06–18 UTC ---
import pandas as pd

times_index = pd.DatetimeIndex(times)
df_temp = pd.DataFrame({'temp': temperature_2m}, index=times_index)

grouped = df_temp.groupby(df_temp.index.date)

y_offset = 0.5  # offset above the line

for day, group in grouped:
    group_day = group[(group.index.hour >= 6) & (group.index.hour <= 18)]
    if group_day.empty:
        continue

    idx_max = group_day['temp'].idxmax()
    temp_max = group_day['temp'].max()

    if idx_max < times[0]:
        continue

    bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=0.8, alpha=0.9)
    ax_temp.text(
        idx_max, temp_max + y_offset, f"{temp_max:.0f}",
        fontsize=9, color='black', ha='center', va='bottom',
        bbox=bbox_props
    )
















# --- Section 5: Wind Gusts ---
def compute_beaufort(knots):
    conditions = [
        (knots >= 0) & (knots < 1),
        (knots >= 1) & (knots < 4),
        (knots >= 4) & (knots < 7),
        (knots >= 7) & (knots < 11),
        (knots >= 11) & (knots < 17),
        (knots >= 17) & (knots < 22),
        (knots >= 22) & (knots < 28),
        (knots >= 28) & (knots < 34),
        (knots >= 34) & (knots < 41),
        (knots >= 41) & (knots < 48),
        (knots >= 48) & (knots < 56),
        (knots >= 56) & (knots < 64),
        knots >= 64
    ]
    values = np.arange(13)
    return np.select(conditions, values)

indices = np.arange(6, len(times), 6)
times_sel = times[indices]
gusts_sel = wind_gusts_10m[indices]
dirs_sel = winddirection_10m[indices]
gusts_knots = gusts_sel / 1.852
gusts_bft = compute_beaufort(gusts_knots)
times_num = mdates.date2num(times_sel)

u = []
v = []
for wd, gs in zip(dirs_sel, gusts_knots):
    to_dir = (wd + 0) % 360
    angle_rad = np.deg2rad((270 - to_dir) % 360)
    u.append(gs * np.cos(angle_rad))
    v.append(gs * np.sin(angle_rad))
u = np.array(u)
v = np.array(v)

y_base = 10.5
spacing_up = 7
spacing_down = 3.5
y_barbs = []
for wd in dirs_sel:
    to_dir = (wd + 0) % 360
    if 0 <= to_dir <= 90 or 270 <= to_dir <= 360:
        y_barbs.append(y_base - spacing_up)
    else:
        y_barbs.append(y_base - spacing_down)
y_barbs = np.array(y_barbs)

times_num_sel = times_num[:-1]
gusts_bft_sel = gusts_bft[:-1]
y_barbs_sel = y_barbs[:-1]
u_sel = u[:-1]
v_sel = v[:-1]

ax_windgust = axs[4]
ax_windgust.clear()
ax_windgust.set_ylim(0, 14)
ax_windgust.set_yticks([])
ax_windgust.set_ylabel('Gusts\n(bft)', fontsize=10)
ax_windgust.grid(axis='y', color='#92A9B6', linestyle='dotted', dashes=(2,5), alpha=0.8)

ax_windgust.barbs(
    times_num_sel,
    y_barbs_sel,
    u_sel, v_sel,
    length=6,
    barbcolor='black',
    linewidth=0.5,
    pivot='tip'
)

for x, bft_val in zip(times_num_sel, gusts_bft_sel):
    if 0 <= bft_val <= 1:
        box_color = 'white'
    elif 2 <= bft_val <= 3:
        box_color = '#C6F6C6'
    elif bft_val == 4:
        box_color = '#FFF5BA'
    elif bft_val == 5:
        box_color = '#FFD580'
    else:
        box_color = '#FFB3B3'
    ax_windgust.text(
        x,
        y_base,
        str(bft_val),
        fontsize=10,
        ha='center',
        va='bottom',
        color='black',
        bbox=dict(facecolor=box_color, edgecolor='none', boxstyle='round,pad=0.3')
    )

# --- Show right spine for all axes ---
for ax in axs:
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)  # optional

# Adjust layout and X-axis formatting
fig.subplots_adjust(top=1.00, bottom=0.15)
axs[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0]))
axs[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: ''))
axs[-1].tick_params(axis='x', which='major', labelsize=9, pad=5)

ticks_00z = [t for t in mdates.date2num(times) if mdates.num2date(t).hour == 0]
labels_00z = [f"{mdates.num2date(t).day}{mdates.num2date(t).strftime('%b').upper()}" for t in ticks_00z]

for tick, label in zip(ticks_00z, labels_00z):
    axs[-1].text(tick, -0.2, label, ha='center', va='top', transform=axs[-1].get_xaxis_transform(which='grid'), fontsize=10)

ax_cloud_secondary_x = axs[0].secondary_xaxis('top')
ax_cloud_secondary_x.set_xticks([])
ax_cloud_secondary_x.set_xticklabels([])
ax_cloud_secondary_x.tick_params(axis='x', which='major', pad=5)

for tick, label in zip(ticks_00z, labels_00z):
    axs[0].text(tick, 1.1, label, ha='center', va='bottom', transform=axs[0].get_xaxis_transform(which='grid'), fontsize=10)




# PLOT IMAGE
run_hour = latest_run_time.strftime("%H")
filename = f"kiato10d_{run_hour}.png"
plt.subplots_adjust(hspace=0.05)
plt.savefig(filename, dpi=96, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)



