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
latitude = 37.98
longitude = 22.76

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
        "geopotential_height_500hPa",
        "geopotential_height_850hPa",
        "freezing_level_height",
        "temperature_850hPa",
        "temperature_500hPa"
    ]),
    "forecast_days": 5,
    "timezone": "UTC",
    "models": "gfs_seamless"
}


# Code that attempts to fetch data with a maximum total wait time of 15 minutes,
# retrying every 60 seconds with a 30-second timeout per request.

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

first_forecast_time = pd.to_datetime(data['hourly']['time'][0])

gfs_generation_time = latest_run_time
print(f"GFS run: {gfs_generation_time:%Y-%m-%d %HZ}")

times_cloud = pd.to_datetime(data['hourly']['time'])
cloud_low = np.array(data['hourly']['cloud_cover_low'])
cloud_mid = np.array(data['hourly']['cloud_cover_mid'])
cloud_high = np.array(data['hourly']['cloud_cover_high'])



time_nums_cloud = mdates.date2num(times_cloud)
dt_cloud = time_nums_cloud[1] - time_nums_cloud[0] if len(time_nums_cloud) > 1 else 1

times = times_cloud.copy()
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
temperature_850 = get_data("temperature_850hPa")
temperature_500 = get_data("temperature_500hPa")
wind_gusts_10m = get_data("wind_gusts_10m")

# New variables 
geopotential_1000 = get_data("geopotential_height_1000hPa")
geopotential_500 = get_data("geopotential_height_500hPa")
geopotential_850 = get_data("geopotential_height_850hPa")
freezing_level = get_data("freezing_level_height") /1000
Z500_1000 = (geopotential_500 - geopotential_1000) / 10  
Z850_1000 = (geopotential_850 - geopotential_1000) / 10  

time_nums = mdates.date2num(times)










# --- Plotting ---
fig, axs = plt.subplots(
    9, 1,
    figsize=(1750 / 160, 1450 / 160), 
    gridspec_kw={'height_ratios': [1.2, 3.2, 0.7, 0.7, 1.5, 0.8, 1, 1, 1]},
    sharex=True
)











# Section 1: Clouds
ax_cloud = axs[0]
ax_cloud.set_facecolor('#00A0FF')
ax_cloud.set_ylim(0, 3)
ax_cloud.set_yticks([0.5, 1.5, 2.5])
ax_cloud.set_yticklabels(['Low', 'Mid', 'High'], fontsize=9, color='black')

for cloud_cover, band_center in zip([cloud_low, cloud_mid, cloud_high], [0.5, 1.5, 2.5]):
    for i in range(len(times_cloud)):
        height = 0.6 * (cloud_cover[i] / 100)
        if height > 0:
            ax_cloud.axhspan(
                band_center - height / 2,
                band_center + height / 2,
                xmin=(time_nums_cloud[i] - dt_cloud / 2 - time_nums_cloud[0]) / (time_nums_cloud[-1] - time_nums_cloud[0]),
                xmax=(time_nums_cloud[i] + dt_cloud / 2 - time_nums_cloud[0]) / (time_nums_cloud[-1] - time_nums_cloud[0]),
                color='white',
                alpha=1.0
            )

ax_cloud.set_title(f"KIATO Init: {latest_run_time:%Y-%m-%d} ({latest_run_time:%HZ})",
                   loc="center", fontsize=14, fontweight='bold', color='black', y=1.9)

ax_cloud.tick_params(axis='y', colors='black')
ax_cloud.set_xlim(time_nums_cloud[0], time_nums_cloud[-1])
ax_cloud.grid(axis='x', color='#92A9B6', linestyle='dotted',dashes=(2, 5),alpha=0.8)

















# Section 2: Humidity & Winds
ax_humidity = axs[1]

# Create meshgrid for plotting
T, P = np.meshgrid(time_nums, pressure_levels)

# Define colormap for humidity
colors = ['#FFFFFF', '#EAF5EA', '#C8D7C8', '#78D778', '#00FF00', '#00BE00']
bounds = [0, 30, 50, 70, 90, 95]
cmap_rh = mcolors.ListedColormap(colors)
norm_rh = mcolors.BoundaryNorm(bounds, cmap_rh.N, extend='max')

# Plot humidity as filled contours
cf = ax_humidity.contourf(
    T, P, humidity,
    levels=bounds,
    cmap=cmap_rh,
    norm=norm_rh,
    extend='max'
)

# Set Y axis (pressure levels)
ax_humidity.set_ylim(1000, 650)
ax_humidity.set_yticks(pressure_levels)
ax_humidity.set_yticklabels(
    ["" if p in (1000, 650) else str(p) for p in pressure_levels],
    fontsize=9
)


# Add gridlines
ax_humidity.grid(axis='x', color='#92A9B6', linestyle='dotted',dashes=(2, 5),alpha=0.8)

# Plot wind barbs every 3 hours, skipping 650 hPa level
indices_3h = np.arange(0, len(times), 3)

for i, p in enumerate(pressure_levels):
    if p == 650:
        continue  # Skip barbs at 650 hPa

    # Convert windspeed from m/s to knots
    ws_knots = windspeed[i][indices_3h] * 0.539957

    # Convert wind direction (meteorological "from") to u/v components
    theta = np.deg2rad(winddirection[i][indices_3h])
    u = -ws_knots * np.sin(theta)
    v = -ws_knots * np.cos(theta)

    # Plot barbs
    ax_humidity.barbs(
        time_nums[indices_3h],
        np.full(len(indices_3h), p),
        u, v,
        length=6,
        linewidth=0.3
    )

# Add humidity contour lines
contour_lines = ax_humidity.contour(
    T, P, humidity,
    levels=bounds[1:],            
    colors='grey',
    linewidths=1,                
    linestyles='--',                
)


# Label contours
ax_humidity.clabel(
    contour_lines,
    fmt='%d',
    fontsize=8,
    inline=True,
    colors='#333333',
)





















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

# Annotate freezing level every 6 hours if < 2 km
offset = (time_nums_3h[1] - time_nums_3h[0]) / 2  # push first label inside
for i in range(0, len(time_nums_3h), 2):  # every 6h (2 x 3h)
    val = freezing_3h[i]
    if val < 2.0: # Select minimum threshold to plot (km)
        x = time_nums_3h[i] + offset if i == 0 else time_nums_3h[i]
        ax_precip.text(
            x, y_max - 0.5, f"{val:.1f}",
            ha='center', va='top',
            fontsize=9, color='blue',
            #fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5)
        )























# Section 6: Pressure 
ax_pressure = axs[5]

# Plot sea level pressure (SLP)
ax_pressure.plot(times, pressure_msl, color='#00A0FF', linewidth=1, label='SLP (hPa)')
ax_pressure.set_ylabel('SLP\n(hPa)', fontsize=9, color='black')
ax_pressure.tick_params(axis='y', labelcolor='black')
ax_pressure.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# Set y-axis limits and ticks for pressure
p_min_rounded = int(np.floor(pressure_msl.min() / 5.0) * 5)
p_max_rounded = int(np.ceil(pressure_msl.max() / 5.0) * 5)
ax_pressure.set_ylim(p_min_rounded, p_max_rounded)
ax_pressure.yaxis.set_major_locator(ticker.MultipleLocator(5))
pressure_ticks = np.arange(p_min_rounded, p_max_rounded + 1, 5)
ax_pressure.set_yticks(pressure_ticks[1:-1])  # exclude first tick























# Section 7: Wind Gusts 10m + Wind Barbs for Windspeed/Direction
ax_windgust = axs[6]  

# wind_gusts_10m 
# Plot wind gusts as solid line
ax_windgust.plot(times, wind_gusts_10m, color='orange', linewidth=1.5, label='Wind Gust (km/h)')

ax_windgust.set_ylabel('Gusts\n(km/h)', fontsize=9, color='black')
ax_windgust.tick_params(axis='y', labelcolor='black')
ax_windgust.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# Set y-axis limits with padding +/- 5, min clipped at 0
gust_min = np.min(wind_gusts_10m)
gust_max = np.max(wind_gusts_10m)
ymin = max(0, gust_min - 5)
ymax = gust_max + 10
ax_windgust.set_ylim(ymin, ymax)
ax_windgust.set_yticks(np.arange(ymin, ymax, 20))

# Plot wind barbs for windspeed_10m & winddirection_10m (NOT gusts)
wind_knots = windspeed_10m * 0.539957
u10 = -wind_knots * np.sin(np.deg2rad(winddirection_10m))
v10 = -wind_knots * np.cos(np.deg2rad(winddirection_10m))

# Place barbs vertically centered between 40%-60% of y-axis range
barb_y = ymin + 0.5 * (ymax - ymin)
time_nums = mdates.date2num(times)
ax_windgust.barbs(time_nums[::3], [barb_y] * len(time_nums[::3]), u10[::3], v10[::3],
                  length=6, linewidth=0.5, color='black', zorder=3)

# Find daily max gusts for labeling
daily_max = {}

for t, gust in zip(times, wind_gusts_10m):
    day = t.date()
    if day not in daily_max or gust > daily_max[day]['value']:
        daily_max[day] = {'value': gust, 'time': t}



# Add boxes with daily max gust values 
for day, info in daily_max.items():
    label_time = pd.Timestamp(info['time'])
    label_val = info['value']

    y_min, y_max = ax_windgust.get_ylim()

    y_offset = 2  # vertical offset for label

    # Try placing label above the line
    text_y = label_val + y_offset
    va = 'bottom'

    # If above the top axis limit, put label below the line
    if text_y > y_max:
        text_y = label_val - y_offset
        va = 'top'

    # Horizontal alignment logic (same as before)
    i = times.get_indexer([label_time])[0]
    if i == 0:
        ha = 'left'
        x_offset = (times[1] - times[0]) / 4
        text_x = label_time + x_offset
    elif i == len(times) - 1:
        ha = 'right'
        x_offset = (times[-1] - times[-2]) / 4
        text_x = label_time - x_offset
    else:
        ha = 'center'
        text_x = label_time

    bbox_props = dict(boxstyle="round,pad=0.3", fc="#FFD8A6", ec="black", lw=0.8, alpha=0.8)

    ax_windgust.text(text_x, text_y, f"{label_val:.0f}", fontsize=8,
                     color='black', ha=ha, va=va, bbox=bbox_props)






























# Section 8: CAPE and Lifted Index
ax_cape = axs[7]
ax_cape.set_ylabel('CAPE\n(J/kg)', fontsize=9, color='black')
ax_cape.tick_params(axis='y', labelcolor='black')

# Filter negative Lifted Index values every 3 hours
time_interval_hours = 1
li_values = lifted_index
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
cape_line, = ax_cape.plot(times, cape, color='black', linestyle='--',label='CAPE (J/kg)', zorder=10)

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
if len(yticks) > 1:
    yticks = yticks[:-1]  # exclude last ytick
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
labels_00z = [mdates.num2date(t).strftime('%d%b').upper() for t in ticks_00z]

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
run_hour = latest_run_time.strftime("%H")
filename = f"kiato{run_hour}.png"
plt.subplots_adjust(hspace=0.05)
plt.savefig(filename, dpi=96, bbox_inches='tight', pad_inches=0)
plt.close(fig)
