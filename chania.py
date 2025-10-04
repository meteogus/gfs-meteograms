import numpy as np
import pandas as pd
import requests
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
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
latitude = 35.51
longitude = 24.02

# API call parameters
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": ",".join([
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "relative_humidity_1000hPa",
        "relative_humidity_950hPa",
        "relative_humidity_900hPa",
        "relative_humidity_850hPa",
        "relative_humidity_800hPa",
        "relative_humidity_750hPa",
        "relative_humidity_700hPa",
        "relative_humidity_650hPa",
        "windspeed_1000hPa",
        "winddirection_1000hPa",
        "windspeed_950hPa",
        "winddirection_950hPa",
        "windspeed_900hPa",
        "winddirection_900hPa",
        "windspeed_850hPa",
        "winddirection_850hPa",
        "windspeed_800hPa",
        "winddirection_800hPa",
        "windspeed_750hPa",
        "winddirection_750hPa",
        "windspeed_700hPa",
        "winddirection_700hPa",
        "windspeed_650hPa",
        "winddirection_650hPa",
        "pressure_msl",
        "windspeed_10m",
        "wind_gusts_10m", 
        "winddirection_10m",
        "cape",
        "lifted_index",
        "precipitation",            
        "showers",         
        "snowfall",
        "geopotential_height_1000hPa",
        "geopotential_height_950hPa",
        "geopotential_height_900hPa",
        "geopotential_height_850hPa",
        "geopotential_height_800hPa",
        "geopotential_height_700hPa",
        "geopotential_height_600hPa",
        "geopotential_height_500hPa",
        "freezing_level_height",
        "temperature_1000hPa",
        "temperature_950hPa",
        "temperature_900hPa",
        "temperature_850hPa",
        "temperature_800hPa",
        "temperature_750hPa",
        "temperature_700hPa",
        "temperature_600hPa",
        "temperature_500hPa"
    ]),
    "forecast_days": 6,
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

pressure_levels = [1000, 950, 900, 850, 800, 750, 700, 650]

def get_data(name):
    return np.array(data['hourly'][name])

humidity = np.array([get_data(f"relative_humidity_{p}hPa") for p in pressure_levels])
windspeed = np.array([get_data(f"windspeed_{p}hPa") for p in pressure_levels])
winddirection = np.array([get_data(f"winddirection_{p}hPa") for p in pressure_levels])
pressure_msl = get_data('pressure_msl')
windspeed_10m = get_data('windspeed_10m')
winddirection_10m = get_data('winddirection_10m')
cape = get_data('cape')
lifted_index = get_data('lifted_index')
precipitation = get_data('precipitation')
showers = get_data('showers')
snowfall = get_data('snowfall')
temperature_1000 = get_data("temperature_1000hPa")
temperature_950 = get_data("temperature_950hPa")
temperature_900 = get_data("temperature_900hPa")
temperature_850 = get_data("temperature_850hPa")
temperature_800 = get_data("temperature_800hPa")
temperature_750 = get_data("temperature_750hPa")
temperature_700 = get_data("temperature_700hPa")
temperature_600 = get_data("temperature_600hPa")
temperature_500 = get_data("temperature_500hPa")
wind_gusts_10m = get_data("wind_gusts_10m")

geopotential_1000 = get_data("geopotential_height_1000hPa")
geopotential_950 = get_data("geopotential_height_950hPa")
geopotential_900 = get_data("geopotential_height_900hPa")
geopotential_850 = get_data("geopotential_height_850hPa")
geopotential_800 = get_data("geopotential_height_800hPa")
geopotential_700 = get_data("geopotential_height_700hPa")
geopotential_600 = get_data("geopotential_height_600hPa")
geopotential_500 = get_data("geopotential_height_500hPa")

freezing_level = get_data("freezing_level_height") 
Z500_1000 = (geopotential_500 - geopotential_1000) / 10
Z850_1000 = (geopotential_850 - geopotential_1000) / 10

geo_heights = np.array([
    geopotential_1000,
    geopotential_950,
    geopotential_900,
    geopotential_850,
    geopotential_800,
    geopotential_700,
    geopotential_600
])  # shape: [levels, time]

pressures = np.array([1000, 950, 900, 850, 800, 700, 600])  # hPa

freeze_h = np.array(freezing_level)  # meters
freeze_p = np.zeros_like(freeze_h, dtype=float)

for i, h in enumerate(freeze_h):
    # Vertical profile of this timestep
    prof_h = geo_heights[:, i]   # heights in meters
    prof_p = pressures           # pressures in hPa (same order)

    # Ensure increasing order in height
    sort_idx = np.argsort(prof_h)
    prof_h = prof_h[sort_idx]
    prof_p = prof_p[sort_idx]

    # Interpolate pressure at freezing height
    if h <= prof_h[0]:
        freeze_p[i] = prof_p[0]   # below lowest model level
    elif h >= prof_h[-1]:
        freeze_p[i] = prof_p[-1]  # above highest model level
    else:
        freeze_p[i] = np.interp(h, prof_h, prof_p)

freezing_level_hpa = freeze_p


# --- FILTER DATA FROM GFS RUN TIME ONWARD (exactly 5 days = 120 hours) ---

start_time = latest_run_time
start_index = np.where(times_cloud >= start_time)[0][0]

# Calculate end_index as start_index + 120 (for 120 hours, assuming hourly data)
end_index = start_index + 120

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

# Slice atmospheric profiles (pressure_levels x time)
humidity = humidity[:, start_index:end_index + 1]
windspeed = windspeed[:, start_index:end_index + 1]
winddirection = winddirection[:, start_index:end_index + 1]

# Slice surface and other variables
pressure_msl = pressure_msl[start_index:end_index + 1]
windspeed_10m = windspeed_10m[start_index:end_index + 1]
winddirection_10m = winddirection_10m[start_index:end_index + 1]
cape = cape[start_index:end_index + 1]
lifted_index = lifted_index[start_index:end_index + 1]
precipitation = precipitation[start_index:end_index + 1]
showers = showers[start_index:end_index + 1]
snowfall = snowfall[start_index:end_index + 1]
temperature_850 = temperature_850[start_index:end_index + 1]
temperature_500 = temperature_500[start_index:end_index + 1]
wind_gusts_10m = wind_gusts_10m[start_index:end_index + 1]

# Slice geopotential and derived thicknesses
geopotential_1000 = geopotential_1000[start_index:end_index + 1]
geopotential_500 = geopotential_500[start_index:end_index + 1]
geopotential_850 = geopotential_850[start_index:end_index + 1]
freezing_level = freezing_level[start_index:end_index + 1]
freezing_level_hpa= freezing_level_hpa[start_index:end_index + 1]
Z500_1000 = Z500_1000[start_index:end_index + 1]
Z850_1000 = Z850_1000[start_index:end_index + 1]

print(f"Data filtered from {times[0]:%Y-%m-%d %HZ} to {times[-1]:%Y-%m-%d %HZ}")



















# --- Plotting ---
fig, axs = plt.subplots(
    9, 1,
    figsize=(1000 / 96, 870 / 96), 
    gridspec_kw={'height_ratios': [1.2, 3.2, 0.7, 0.7, 1.5, 0.8, 1, 1, 1]},
    sharex=True
)











# Section 1: Clouds
ax_cloud = axs[0]
ax_cloud.set_facecolor('#00A0FF')
ax_cloud.set_ylim(0, 3)
ax_cloud.set_yticks([0.5, 1.5, 2.5])
ax_cloud.set_yticklabels(['Low', 'Mid', 'High'], fontsize=9, color='black')

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
    f"CHANIA Init: {latest_run_time:%Y-%m-%d} ({latest_run_time:%HZ})",
    loc="center", fontsize=14, fontweight='bold', color='black', y=1.9
)

ax_cloud.tick_params(axis='y', colors='black')
ax_cloud.set_xlim(time_nums[0], time_nums[-1])
ax_cloud.grid(axis='x', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)
















# --- Section 2: Humidity & Winds (complete, no NaNs; capped display at 650 hPa) ---
ax_humidity = axs[1]

# Pressure levels (surface → top of section). Keep this order.
pressure_levels = np.array([1000, 950, 900, 850, 800, 750, 700, 650])

# --- Prepare 3-hourly time indices and numeric times ---
indices_3h = np.arange(0, len(times), 3)
times_3h = [times[i] for i in indices_3h]
time_nums_3h = mdates.date2num(times_3h)
time_nums_all = mdates.date2num(times)

# --- Display clamp: force any above-top values to exactly top_limit for plotting alignment ---
top_limit = 650
# Values < top_limit mean 'above the top' because pressure decreases with altitude.
p_display = np.where(freezing_level_hpa < top_limit, top_limit, freezing_level_hpa)


# --- Meshgrid for humidity plotting ---
T, P = np.meshgrid(time_nums_all, pressure_levels)

# --- Humidity filled contours ---
colors = ['#FFFFFF', '#EAF5EA', '#C8D7C8', '#78D778', '#00FF00', '#00BE00']
bounds = [0, 30, 50, 70, 90, 95]
cmap_rh = mcolors.ListedColormap(colors)
norm_rh = mcolors.BoundaryNorm(bounds, cmap_rh.N, extend='max')
cf = ax_humidity.contourf(T, P, humidity, levels=bounds, cmap=cmap_rh, norm=norm_rh, extend='max')

# --- Y-axis: 1000 (bottom) to 650 (top); hide labels for extremes ---
yticks = [p for p in pressure_levels if p >= 650]
ax_humidity.set_ylim(1000, 650)
ax_humidity.set_yticks(yticks)
ax_humidity.set_yticklabels(["" if p in (1000, 650) else str(p) for p in yticks], fontsize=9)

# Gridlines
ax_humidity.grid(axis='x', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# --- Wind barbs (skip level 650) ---
for i, p in enumerate(pressure_levels):
    if p == 650:
        continue
    ws_knots = windspeed[i][indices_3h] * 0.539957
    theta = np.deg2rad(winddirection[i][indices_3h])
    u = -ws_knots * np.sin(theta)
    v = -ws_knots * np.cos(theta)
    ax_humidity.barbs(time_nums_3h, np.full(len(indices_3h), p), u, v, length=6, linewidth=0.3)

# --- Contour lines on top ---
contour_lines = ax_humidity.contour(T, P, humidity, levels=bounds[1:], colors='grey', linewidths=1, linestyles='--')
ax_humidity.clabel(contour_lines, fmt='%d', fontsize=8, inline=True, colors='#333333')

freezing_level_hpa_3h = freezing_level_hpa[::3]  # pick every 3rd element
ax_humidity.plot(
    time_nums_3h, freezing_level_hpa_3h,
    color='white', linewidth=0.8, solid_capstyle='round', zorder=30,
    path_effects=[
        pe.Stroke(linewidth=1.8, foreground='black'),
        pe.Stroke(linewidth=1.8, foreground='white'),
        pe.Stroke(linewidth=1.8, foreground='black'),
        pe.Normal()
    ]
)

# --- Draw "0" boxes at 00Z only when freezing level is visible inside section (>= top_limit) ---
bbox_props = dict(boxstyle="round,pad=0.02", fc="black", ec="black", lw=1)
# Ensure final ylim is used for nudging
y0, y1 = ax_humidity.get_ylim()
vis_ymin, vis_ymax = min(y0, y1), max(y0, y1)
edge_margin = (vis_ymax - vis_ymin) * 0.03

for i, dt in enumerate(times_3h):
    if getattr(dt, "hour", None) == 00 and getattr(dt, "minute", None) in (0, None):
        pv = freezing_level_hpa_3h[i]
        if pv >= top_limit:  # visible inside section
            y_val = pv
            # nudge down if too close to top edge
            if y_val >= vis_ymax - edge_margin:
                y_val = vis_ymax - edge_margin
            ax_humidity.text(mdates.date2num(dt), y_val, "0",
                             fontsize=8, color='white', ha='center', va='center',
                             bbox=bbox_props, zorder=50)



# --- Print freezing level at 3-hourly steps ---
for dt, p in zip(times_3h, freezing_level_hpa_3h):
    print(f"{dt}: {p:.1f} hPa")






















# Section 3: Temperature 500 hPa
ax_temp500 = axs[2]  

# Plot temperature 500 hPa
line1 = ax_temp500.plot(times, temperature_500, label='Temp 500 hPa (°C)', color='blue', linewidth=1.2)
ax_temp500.set_ylabel('T500\n(°C)', fontsize=9, color='black')
ax_temp500.tick_params(axis='y', labelcolor='black')

# Set y-limits: adjusted min-1 and max+1
t500_min = np.min(temperature_500)
t500_max = np.max(temperature_500)
t500_lower = int(np.floor(t500_min)) - 1
t500_upper = int(np.ceil(t500_max)) + 1
ax_temp500.set_ylim(t500_lower, t500_upper)

# Remove y-ticks completely
ax_temp500.set_yticks([])

# Calculate horizontal offset: ~1/4 of time interval
x_offset = (times[1] - times[0]) / 4

# Add adaptive labels at every 6 hours (00Z, 06Z, 12Z, 18Z)
for i, (t, temp) in enumerate(zip(times, temperature_500)):
    if t.hour % 6 == 0:
        # Decide if label goes above or below
        if temp < (t500_lower + t500_upper) / 2:
            y_pos = temp + 0.5
            va = 'bottom'
        else:
            y_pos = temp - 0.5
            va = 'top'

        # Box styling for 00Z times
        if t.hour == 0:
            bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=0.8, alpha=0.7)
        else:
            bbox_props = None

        # Adjust horizontal alignment for first and last points
        if i == 0:
            ax_temp500.text(t + x_offset, y_pos, f"{temp:.0f}",
                            fontsize=8, color='black', ha='left', va=va, bbox=bbox_props)
        elif i == len(times) - 1:
            ax_temp500.text(t - x_offset, y_pos, f"{temp:.0f}",
                            fontsize=8, color='black', ha='right', va=va, bbox=bbox_props)
        else:
            ax_temp500.text(t, y_pos, f"{temp:.0f}",
                            fontsize=8, color='black', ha='center', va=va, bbox=bbox_props)

ax_temp500.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)













# Section 4: Temperature 850 hPa
ax_temp850 = axs[3]  # Assuming axs[3] is for T850

# Plot temperature 850 hPa
line2 = ax_temp850.plot(times, temperature_850, label='Temp 850 hPa (°C)', color='red', linewidth=1.2)
ax_temp850.set_ylabel('T850\n(°C)', fontsize=9, color='black')
ax_temp850.tick_params(axis='y', labelcolor='black')

# Set y-limits: adjusted min-1 and max+1
t850_min = np.min(temperature_850)
t850_max = np.max(temperature_850)
t850_lower = int(np.floor(t850_min)) - 1
t850_upper = int(np.ceil(t850_max)) + 1
ax_temp850.set_ylim(t850_lower, t850_upper)

# Remove y-ticks completely
ax_temp850.set_yticks([])

# Calculate horizontal offset: ~1/4 of time interval
x_offset = (times[1] - times[0]) / 4

# Add adaptive labels at every 6 hours (00Z, 06Z, 12Z, 18Z)
for i, (t, temp) in enumerate(zip(times, temperature_850)):
    if t.hour % 6 == 0:
        # Decide if label goes above or below
        if temp < (t850_lower + t850_upper) / 2:
            y_pos = temp + 0.5
            va = 'bottom'
        else:
            y_pos = temp - 0.5
            va = 'top'

        # Box styling for 00Z times
        if t.hour == 0:
            bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=0.8, alpha=0.7)
        else:
            bbox_props = None

        # Adjust horizontal alignment for first and last points
        if i == 0:
            ax_temp850.text(t + x_offset, y_pos, f"{temp:.0f}",
                           fontsize=8, color='black', ha='left', va=va, bbox=bbox_props)
        elif i == len(times) - 1:
            ax_temp850.text(t - x_offset, y_pos, f"{temp:.0f}",
                           fontsize=8, color='black', ha='right', va=va, bbox=bbox_props)
        else:
            ax_temp850.text(t, y_pos, f"{temp:.0f}",
                           fontsize=8, color='black', ha='center', va=va, bbox=bbox_props)


ax_temp850.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)












# Section 5: Precipitation with Freezing Level values (every 6h, first inside section)
ax_precip = axs[4]
bar_width = (time_nums[1] - time_nums[0]) * 1.8
bar_width_showers = (time_nums[1] - time_nums[0]) * 0.9

# Convert to numpy arrays
rain = np.array(precipitation)          # rain (mm per hour)
showers_arr = np.array(showers)         # showers (mm per hour)
snowfall_arr = np.array(snowfall)       # snowfall (mm per hour)
freezing_arr = np.array(freezing_level) # freezing level (m)
time_arr = np.array(time_nums)

# Reshape into 3h blocks and sum
n = (len(rain) // 3) * 3
rain_3h = rain[:n].reshape(-1, 3).sum(axis=1)
showers_3h = showers_arr[:n].reshape(-1, 3).sum(axis=1)
snowfall_3h = snowfall_arr[:n].reshape(-1, 3).sum(axis=1)
freezing_3h_km = (freezing_arr[:n].reshape(-1, 3).mean(axis=1)) / 1000  # average freezing level
time_nums_3h = time_arr[:n].reshape(-1, 3)[:, 0]  # timestamp of first hour in block

# Plot precipitation bars
ax_precip.bar(time_nums_3h, rain_3h, width=bar_width, color='#20D020', alpha=1.0, label='Rain')
ax_precip.bar(time_nums_3h, showers_3h, width=bar_width_showers, color='#FA3C3C', alpha=1.0, label='Showers')
ax_precip.bar(time_nums_3h, snowfall_3h, width=bar_width, color='#4040FF', alpha=1.0, label='Snowfall')

# Plot Freezing level line
# ax_frlabel = ax_precip.twinx()
# ax_frlabel.plot(time_nums_3h, freezing_3h, color='#0072B2', linestyle='-', linewidth=0.7)
# ax_frlabel.set_yticks([])

# Y-axis setup
ax_precip.set_ylabel('Precip.\n(mm)', fontsize=9, color='black')
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
ax_frlabel.set_ylabel("Fr.Level\n(km)", fontsize=9, color='blue', rotation=90)
ax_frlabel.yaxis.set_label_position("right")
ax_frlabel.yaxis.tick_right()
ax_frlabel.tick_params(right=False, labelright=False)  # Hide right ticks and labels

# Annotate freezing level every 6 hours if < 2 km
offset = (time_nums_3h[1] - time_nums_3h[0]) / 2  # push first label inside
for i in range(0, len(time_nums_3h), 2):  # every 6h (2 x 3h)
    val = freezing_3h_km[i]
    if val <= 1.7: # Select minimum threshold to plot (km)
        x = time_nums_3h[i] + offset if i == 0 else time_nums_3h[i]
        ax_precip.text(
            x, y_max - 0.5, f"{val:.1f}",
            ha='center', va='top',
            fontsize=9, color='blue',
            #fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5)
        )























# --- Section 6: Pressure ---
ax_pressure = axs[5]

# Plot sea level pressure (SLP)
ax_pressure.plot(times, pressure_msl, color='#00A0FF', linewidth=1, label='SLP (hPa)')
ax_pressure.set_ylabel('SLP\n(hPa)', fontsize=9, color='black')
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

























# --- Section 7: Wind Gusts and Beaufort Scale Visualization ---

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
indices = np.arange(3, len(times), 3)

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
ax_windgust = axs[6]
ax_windgust.clear()
ax_windgust.set_ylim(0, 14)
ax_windgust.set_yticks([])
ax_windgust.set_ylabel('Gusts\n(bft)', fontsize=9)
ax_windgust.grid(axis='y', color='#92A9B6', linestyle='dotted', dashes=(2,5), alpha=0.8)

# Plot wind barbs at selected times and heights (excluding first and last)
ax_windgust.barbs(
    times_num_sel,
    y_barbs_sel,
    u_sel, v_sel,
    length=5.5,
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
        fontsize=10,
        ha='center',
        va='bottom',
        color='black',
        bbox=dict(facecolor=box_color, edgecolor='none', boxstyle='round,pad=0.3')
    )




























































# Section 8: CAPE and Lifted Index
ax_cape = axs[7]
ax_cape.set_ylabel('CAPE\n(J/kg)', fontsize=9, color='#007F7F')
ax_cape.tick_params(axis='y', labelcolor='#007F7F')

# Filter negative Lifted Index values every 3 hours
time_interval_hours = 1
li_values = lifted_index
first_forecast_time = times[0]
gfs_run_time = first_forecast_time

li_times = [gfs_run_time + pd.Timedelta(hours=i * time_interval_hours) for i in range(len(li_values))]

neg_li_times = []
neg_li_values = []

for t, v in zip(li_times, li_values):
    if v <= 0 and t.hour % 3 == 0:
        neg_li_times.append(t)
        neg_li_values.append(v)

ax_li = ax_cape.twinx()

# Bars axis below CAPE axis
ax_li.set_zorder(1)            # Bars axis lower
ax_cape.set_zorder(2)          # CAPE axis above

# Transparent background on CAPE axis so bars show through
ax_cape.patch.set_alpha(0)

# Plot bars on ax_li with lower zorder
bars = ax_li.bar(neg_li_times, neg_li_values, color='#F08228', width=0.08, align='center', zorder=1)

# Plot the red CAPE line with very high zorder on ax_cape (always on top)
cape_line, = ax_cape.plot(times, cape, color='#007F7F', label='CAPE (J/kg)', zorder=10)

# Other elements (grid, zero line)
ax_cape.grid(axis='both', color='#92A9B6', linestyle='dotted',dashes=(2, 5),alpha=0.7, zorder=0)

max_cape = cape.max()
ymax = max_cape + 200

if max_cape < 1000:
    step = 200
else:
    step = 400

ax_cape.set_ylim(0, ymax)
yticks = np.arange(step, ymax + 1, step)
ax_cape.set_yticks(yticks)


# Axes labels and ticks unchanged
ax_li.set_ylabel('Lifted\n index', fontsize=9, color='#F08228')
ax_li.tick_params(axis='y', labelcolor='#F08228')
ax_li.set_ylim(0, -6)
ax_li.set_yticks(np.arange(-2, -8, -2))



















# Section 9: Z500_1000 (left y-axis) and Z850_1000 (right y-axis)
ax_Z500_1000 = axs[8]  # Base axis for Z500_1000

# Plot Z500_1000 on primary y-axis (left)
ax_Z500_1000.plot(
    times, Z500_1000, 
    color='purple', linewidth=1.5, label='Z500_1000 (500–1000 hPa)'
)

# Set Z500_1000 y-axis limits and ticks
Z500_1000_min = np.min(Z500_1000)
Z500_1000_max = np.max(Z500_1000)
Z500_1000_lower = 5 * np.floor(Z500_1000_min / 5)
Z500_1000_upper = 5 * np.ceil(Z500_1000_max / 5)
Z500_1000_ticks = np.arange(Z500_1000_lower, Z500_1000_upper + 1, 5)  # every 5 dm

# Exclude first and last ticks
if len(Z500_1000_ticks) > 1:
    Z500_1000_ticks = Z500_1000_ticks[:-1]

ax_Z500_1000.set_ylim(Z500_1000_lower, Z500_1000_upper)
ax_Z500_1000.set_yticks(Z500_1000_ticks)

# Label and style for Z500_1000 axis (left)
ax_Z500_1000.set_ylabel("Z500-Z1000\n(dm)", fontsize=9, color='purple')
ax_Z500_1000.tick_params(axis='y', labelcolor='purple')
ax_Z500_1000.yaxis.set_label_position("left")
ax_Z500_1000.yaxis.tick_left()


# Create secondary y-axis for Z850_1000 (right side)
ax_Z850_1000 = ax_Z500_1000.twinx()

# Plot Z850_1000 on right axis
ax_Z850_1000.plot(
    times, Z850_1000, 
    color='green', linewidth=1.5, linestyle='--', label='Z850_1000 (850–1000 hPa)'
)

# Set Z850_1000 y-axis limits and ticks 
Z850_1000_min = np.min(Z850_1000)
Z850_1000_max = np.max(Z850_1000)
Z850_1000_lower = 2 * np.floor(Z850_1000_min / 2)
Z850_1000_upper = 2 * np.ceil(Z850_1000_max / 2)
Z850_1000_ticks = np.arange(Z850_1000_lower, Z850_1000_upper + 0.1, 2) # every n dm (n=last number)

# Exclude last tick 
if len(Z850_1000_ticks) > 1:
    Z850_1000_ticks = Z850_1000_ticks[:-1]

ax_Z850_1000.set_ylim(Z850_1000_lower, Z850_1000_upper)
ax_Z850_1000.set_yticks(Z850_1000_ticks)

# Label and style for Z850_1000 axis (right)
ax_Z850_1000.set_ylabel("Z850-Z1000\n(dm)", fontsize=9, color='green')
ax_Z850_1000.tick_params(axis='y', labelcolor='green')
ax_Z850_1000.yaxis.set_label_position("right")
ax_Z850_1000.yaxis.tick_right()

# Grid styling (only on primary axis)
ax_Z500_1000.grid(
    which='both', axis='both',
    color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8
)
































# Adjust layout and X-axis formatting

fig.subplots_adjust(top=1.00, bottom=0.15)

axs[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
axs[-1].xaxis.set_major_formatter(FuncFormatter(
    lambda x, pos: mdates.num2date(x).strftime('%HZ')
))
axs[-1].tick_params(axis='x', which='major', labelsize=9, pad=5)

ticks_00z = [t for t in mdates.date2num(times) if mdates.num2date(t).hour == 0]

labels_00z = [f"{mdates.num2date(t).day}{mdates.num2date(t).strftime('%b').upper()}" for t in ticks_00z]

for tick, label in zip(ticks_00z, labels_00z):
    axs[-1].text(
        tick, -0.4,
        label, ha='center', va='top',
        transform=axs[-1].get_xaxis_transform(which='grid'),
        fontsize=10
    )

ax_cloud_secondary_x = axs[0].secondary_xaxis('top')
ax_cloud_secondary_x.set_xticks(
    [t for t in mdates.date2num(times) if mdates.num2date(t).hour in [0, 12]]
)
ax_cloud_secondary_x.set_xticklabels(
    [mdates.num2date(t).strftime('%HZ') for t in mdates.date2num(times)
     if mdates.num2date(t).hour in [0, 12]],
    fontsize=9
)
ax_cloud_secondary_x.tick_params(axis='x', which='major', pad=5)

for tick, label in zip(ticks_00z, labels_00z):
    axs[0].text(
        tick, 1.35,
        label, ha='center', va='bottom',
        transform=axs[0].get_xaxis_transform(which='grid'),
        fontsize=10
    )

# --- NEW: Add day names (MON, TUE, etc.) above the date labels on top subplot ---

day_labels = [mdates.num2date(t).strftime('%a').upper() for t in ticks_00z]

for tick, day in zip(ticks_00z, day_labels):
    axs[0].text(
        tick, 1.55,        # Slightly above your date labels (which are at 1.35)
        day,
        ha='center', va='bottom',
        transform=axs[0].get_xaxis_transform(which='grid'),
        fontsize=9,
        fontweight='bold',
        color='black'       # <-- added this line
    )















# PLOT IMAGE
plt.subplots_adjust(hspace=0.05)
plt.savefig("chania.png", dpi=96, bbox_inches='tight', pad_inches=0)
plt.close(fig)















