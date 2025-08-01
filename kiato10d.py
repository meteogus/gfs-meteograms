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
        "freezing_level_height",
    ]),
    "forecast_days": 11,
    "timezone": "UTC",
    "models": "gfs_seamless"
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
freezing_level = get_data("freezing_level_height") / 1000











from datetime import timedelta

# --- FILTER DATA FROM GFS RUN TIME ONWARD ---

start_time = latest_run_time
start_index = np.where(times_cloud >= start_time)[0][0]

# Calculate end_index as start_index + 264 (for 264 hours, assuming hourly data)
end_index = start_index + 264

# Prevent going beyond array length
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

print(f"Data filtered from {times[0]:%Y-%m-%d %HZ} to {times[-1]:%Y-%m-%d %HZ}")



















# --- Plotting ---
fig, axs = plt.subplots(
    4, 1,
    figsize=(1500 / 96, 600 / 96), 
    gridspec_kw={'height_ratios': [1.2, 1, 1, 1]},
    sharex=True
)











# Section 1: Clouds
ax_cloud = axs[0]
ax_cloud.set_facecolor('#00A0FF')
ax_cloud.set_ylim(0, 3)
ax_cloud.set_yticks([0.5, 1.5, 2.5])
ax_cloud.set_yticklabels(['Low', 'Mid', 'High'], fontsize=12, color='black')

# Convert times_cloud to matplotlib date numbers once (should be sliced times)
time_nums = mdates.date2num(times)

# Calculate dt_cloud as time step in days
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
    loc="center", fontsize=14, fontweight='bold', color='black', y=1.4
)

ax_cloud.tick_params(axis='y', colors='black')
ax_cloud.set_xlim(time_nums[0], time_nums[-1])
ax_cloud.grid(axis='x', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)




































# Section 2: Precipitation with Freezing Level values (every 6h, first inside section)
ax_precip = axs[1]
bar_width = (time_nums[1] - time_nums[0]) * 1.8
bar_width_showers = (time_nums[1] - time_nums[0]) * 0.9

# 3-hourly data
time_nums_3h = time_nums[::3]
rain_3h = precipitation[::3]
showers_3h = showers[::3]
snowfall_3h = snowfall[::3]
freezing_3h = freezing_level[::3]

# Plot precipitation bars
ax_precip.bar(time_nums_3h, rain_3h, width=bar_width, color='#20D020', alpha=1.0, label='Rain')
ax_precip.bar(time_nums_3h, showers_3h, width=bar_width_showers, color='#FA3C3C', alpha=1.0, label='Showers')
ax_precip.bar(time_nums_3h, snowfall_3h, width=bar_width, color='#4040FF', alpha=1.0, label='Snowfall')

# Plot Freezing level line
# ax_frlabel = ax_precip.twinx()
# ax_frlabel.plot(time_nums_3h, freezing_3h, color='#0072B2', linestyle='-', linewidth=0.7)
# ax_frlabel.set_yticks([])

# Y-axis setup
ax_precip.set_ylabel('Precip.\n(mm)', fontsize=12, color='black')
ax_precip.tick_params(axis='y', labelcolor='black')
ax_precip.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

max_precip = max(np.max(rain_3h), np.max(showers_3h), np.max(snowfall_3h))
y_step = 2 if max_precip <= 10 else 5 if max_precip <= 30 else 10
y_max = y_step * np.ceil((max_precip + 2) / y_step)
ax_precip.set_ylim(0, y_max)
ax_precip.set_yticks(np.arange(y_step, y_max + y_step, y_step))

# Right-side label for freezing level
ax_frlabel = ax_precip.twinx()
ax_frlabel.set_ylim(ax_precip.get_ylim())
ax_frlabel.set_ylabel("Fr.Level\n(km)", fontsize=12, color='blue', rotation=90)
ax_frlabel.yaxis.set_label_position("right")
ax_frlabel.yaxis.tick_right()

# Annotate freezing level every 6 hours if < 2 km
offset = (time_nums_3h[1] - time_nums_3h[0]) / 2  # push first label inside
for i in range(0, len(time_nums_3h), 2):  # every 6h (2 x 3h)
    val = freezing_3h[i]
    if val < 2.0: # Select minimum threshold to plot (km)
        x = time_nums_3h[i] + offset if i == 0 else time_nums_3h[i]
        ax_precip.text(
            x, y_max - 0.5, f"{val:.1f}",
            ha='center', va='top',
            fontsize=12, color='blue',
            #fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5)
        )























# --- Section 3: Pressure ---
ax_pressure = axs[2]

# Plot sea level pressure (SLP)
ax_pressure.plot(times, pressure_msl, color='#00A0FF', linewidth=1, label='SLP (hPa)')
ax_pressure.set_ylabel('SLP\n(hPa)', fontsize=12, color='black')
ax_pressure.tick_params(axis='y', labelcolor='black')
ax_pressure.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# Set y-axis limits rounded to nearest 5
pmin_rounded = int(np.floor(pressure_msl.min() / 5.0) * 5)
pmax_rounded = int(np.ceil(pressure_msl.max() / 5.0) * 5)

# Choose tick step based on pressure range
p_range = pmax_rounded - pmin_rounded
if p_range > 20:
    step = 10
else:
    step = 5

# Set y-axis limits and ticks
ax_pressure.set_ylim(pmin_rounded, pmax_rounded)
yticks = np.arange(pmin_rounded, pmax_rounded + 1, step)
ax_pressure.set_yticks(yticks)
























# --- Section 4: Wind Gusts and Beaufort Scale Visualization ---

def compute_beaufort(knots):
    conditions = [
        (knots >= 0) & (knots < 1),      # 0 Bft
        (knots >= 1) & (knots < 4),      # 1 Bft
        (knots >= 4) & (knots < 7),      # 2 Bft
        (knots >= 7) & (knots < 11),     # 3 Bft
        (knots >= 11) & (knots < 17),    # 4 Bft
        (knots >= 17) & (knots < 22),    # 5 Bft
        (knots >= 22) & (knots < 28),    # 6 Bft
        (knots >= 28) & (knots < 34),    # 7 Bft
        (knots >= 34) & (knots < 41),    # 8 Bft
        (knots >= 41) & (knots < 48),    # 9 Bft
        (knots >= 48) & (knots < 56),    # 10 Bft
        (knots >= 56) & (knots < 64),    # 11 Bft
        knots >= 64                      # 12 Bft
    ]

    values = np.arange(13)
    return np.select(conditions, values)

# Select every 3rd data point starting from index 3 to reduce clutter
indices = np.arange(6, len(times), 6)

times_sel = times[indices]
gusts_sel = wind_gusts_10m[indices]          # gusts in km/h from data
dirs_sel = winddirection_10m[indices]        # wind direction in degrees

# --- UNIT CONVERSION ---
# Convert gusts from km/h to knots for Beaufort scale and barbs
gusts_knots = gusts_sel / 1.852

# Compute Beaufort scale values (0 to 12)
gusts_bft = compute_beaufort(gusts_knots)

# Convert times to matplotlib date format for plotting
times_num = mdates.date2num(times_sel)

# Calculate wind barb vector components (u,v) from direction and speed in knots
u = []
v = []
for wd, gs in zip(dirs_sel, gusts_knots):
    # Wind direction is meteorological (where wind comes FROM)
    # Convert to direction TO for vector components
    to_dir = (wd + 0) % 360
    # Angle for plotting (matplotlib uses x=0° east, angles CCW)
    angle_rad = np.deg2rad((270 - to_dir) % 360)
    u.append(gs * np.cos(angle_rad))
    v.append(gs * np.sin(angle_rad))
u = np.array(u)
v = np.array(v)

# Position barbs slightly above or below baseline depending on wind direction quadrant
y_base = 9
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

# Exclude first and last points to avoid first and last Beaufort boxes and barbs
times_num_sel = times_num[:-1]
gusts_bft_sel = gusts_bft[:-1]
y_barbs_sel = y_barbs[:-1]
u_sel = u[:-1]
v_sel = v[:-1]

# Clear and setup wind gust subplot
ax_windgust = axs[3]
ax_windgust.clear()
ax_windgust.set_ylim(0, 14)
ax_windgust.set_yticks([])
ax_windgust.set_ylabel('Gusts\n(bft)', fontsize=12)
ax_windgust.grid(axis='y', color='#92A9B6', linestyle='dotted', dashes=(2,5), alpha=0.8)

# Plot wind barbs at selected times and heights (excluding first and last)
ax_windgust.barbs(
    times_num_sel,
    y_barbs_sel,
    u_sel, v_sel,
    length=7,
    barbcolor='black',
    linewidth=0.5,
    pivot='tip'
)

# Plot Beaufort numbers as colored boxes exactly at the barbs (excluding first and last)
for x, bft_val in zip(times_num_sel, gusts_bft_sel):
    if 0 <= bft_val <= 1:
        box_color = 'white'          # Calm sea (Beaufort 0–1)
    elif 2 <= bft_val <= 3:
        box_color = '#C6F6C6'        # Light green – Light breeze (Beaufort 2–3)
    elif bft_val == 4:
        box_color = '#FFF5BA'        # Light yellow – Moderate breeze (Beaufort 4)
    elif bft_val == 5:
        box_color = '#FFD580'        # Light orange – Fresh breeze / near strong (Beaufort 5)
    else:
        box_color = '#FFB3B3'        # Light red – Strong breeze or higher (Beaufort 6+)
    
    ax_windgust.text(
        x,
        y_base,
        str(bft_val),
        fontsize=14,
        ha='center',
        va='bottom',
        color='black',
        bbox=dict(facecolor=box_color, edgecolor='none', boxstyle='round,pad=0.3')
    )




























# Adjust layout and X-axis formatting
fig.subplots_adjust(top=1.00, bottom=0.15)

# Set major ticks at 00Z to mark daily ticks
axs[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0]))

# Suppress hour labels (like 00Z)
axs[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: ''))
axs[-1].tick_params(axis='x', which='major', labelsize=9, pad=5)

# Extract ticks at 00Z and format as date labels (e.g., 06AUG)
ticks_00z = [t for t in mdates.date2num(times) if mdates.num2date(t).hour == 0]
date_labels = [mdates.num2date(t).strftime('%d%b').upper() for t in ticks_00z]

# Add date labels below bottom subplot
for tick, label in zip(ticks_00z, date_labels):
    axs[-1].text(
        tick, -0.2,
        label,
        ha='center', va='top',
        transform=axs[-1].get_xaxis_transform(which='grid'),
        fontsize=12
    )

# Set up the top secondary axis with no tick marks or labels
ax_cloud_secondary_x = axs[0].secondary_xaxis('top')
ax_cloud_secondary_x.set_xticks([])
ax_cloud_secondary_x.set_xticklabels([])
ax_cloud_secondary_x.tick_params(axis='x', which='major', pad=5)

# Add date labels above the top subplot
for tick, label in zip(ticks_00z, date_labels):
    axs[0].text(
        tick, 1.1,
        label,
        ha='center', va='bottom',
        transform=axs[0].get_xaxis_transform(which='grid'),
        fontsize=12
    )






# PLOT IMAGE
plt.subplots_adjust(hspace=0.05)
plt.savefig("kiato10d.png", dpi=96, bbox_inches='tight', pad_inches=0)
plt.close(fig)
