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

# ECMWF runs at 00Z and 12Z
ecmwf_run_hours = [0, 12]

# Find the most recent ECMWF run
latest_run_hour = max([h for h in ecmwf_run_hours if h <= now_utc.hour])
latest_run_time = now_utc.replace(hour=latest_run_hour, minute=0, second=0, microsecond=0)

# If current time is earlier than first run (00Z), go back one day
if now_utc.hour < ecmwf_run_hours[0]:
    latest_run_time -= timedelta(days=1)

print(f"Latest ECMWF run: {latest_run_time:%Y-%m-%d %HZ}")

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
        "cape",
        "precipitation",            
        "showers",         
        "snowfall",
        "temperature_2m",
        "relative_humidity_2m"
    ]),
    "forecast_days": 6,
    "timezone": "UTC",
    "models": "ecmwf_ifs"
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
    return np.array(data['hourly'][name], dtype=float)

pressure_msl = get_data('pressure_msl')
windspeed_10m = get_data('windspeed_10m')
winddirection_10m = get_data('winddirection_10m')
cape = get_data('cape')
precipitation = get_data('precipitation')
showers = get_data('showers')
snowfall = get_data('snowfall')
wind_gusts_10m = get_data("wind_gusts_10m")
temperature_2m = get_data('temperature_2m')
relative_humidity_2m = get_data('relative_humidity_2m')




# --- FILTER DATA FROM GFS RUN TIME ONWARD (exactly 5 days = 120 hours) ---

start_time = latest_run_time
start_index = np.where(times_cloud >= start_time)[0][0]

# Calculate end_index as start_index + 144 (for 144 hours, assuming hourly data)
end_index = start_index + 144

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
cape = cape[start_index:end_index + 1]
precipitation = precipitation[start_index:end_index + 1]
showers = showers[start_index:end_index + 1]
snowfall = snowfall[start_index:end_index + 1]
wind_gusts_10m = wind_gusts_10m[start_index:end_index + 1]
temperature_2m = temperature_2m[start_index:end_index + 1]
relative_humidity_2m = relative_humidity_2m[start_index:end_index + 1]

print(f"Data filtered from {times[0]:%Y-%m-%d %HZ} to {times[-1]:%Y-%m-%d %HZ}")









fig, axs = plt.subplots(
    7, 1,
    figsize=(1000 / 96, 857 / 96),
    gridspec_kw={'height_ratios': [0.15, 0.23, 0.2, 0.2, 0.2, 0.2, 0.2]},
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

# Decide run label based on UTC time
if now_utc.hour < 12:
    run_label = "00Z"
else:
    run_label = "12Z"

ax_cloud.set_title(
    f"KIATO ECMWF ({run_label})",
    loc="center", fontsize=14, fontweight='bold', color='black', y=1.8
)

ax_cloud.tick_params(axis='y', colors='black')
ax_cloud.set_xlim(time_nums[0], time_nums[-1])
ax_cloud.grid(axis='x', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)





















# --- Section 2: Precipitation (every 3h) ---

ax_precip = axs[1]

# Bar width setup (unchanged)
bar_width = (time_nums[1] - time_nums[0]) * 1.8
bar_width_showers = (time_nums[1] - time_nums[0]) * 0.9

# Convert to numpy arrays
rain = np.array(precipitation)          # rain (mm per hour)
showers_arr = np.array(showers)         # showers (mm per hour)
snowfall_arr = np.array(snowfall)       # snowfall (mm per hour)
cape_arr = np.array(cape)               # CAPE values
time_arr = np.array(time_nums)

# Reshape into 3h blocks
n = (len(rain) // 3) * 3
rain_3h = rain[:n].reshape(-1, 3).sum(axis=1)
showers_3h = showers_arr[:n].reshape(-1, 3).sum(axis=1)
snowfall_3h = snowfall_arr[:n].reshape(-1, 3).sum(axis=1)
cape_3h = cape_arr[:n].reshape(-1, 3).max(axis=1)  # max CAPE 3h block
time_nums_3h = time_arr[:n].reshape(-1, 3)[:, 0]   # timestamp of first hour

# --- Plot precipitation bars (unchanged) ---
ax_precip.bar(time_nums_3h, rain_3h, width=bar_width, color='#20D020', alpha=1.0, label='Rain')
ax_precip.bar(time_nums_3h, showers_3h, width=bar_width_showers, color='#FA3C3C', alpha=1.0, label='Showers')
ax_precip.bar(time_nums_3h, snowfall_3h, width=bar_width, color='#4040FF', alpha=1.0, label='Snowfall')

# Y-axis setup
ax_precip.set_ylabel('Precip.\n(mm)', fontsize=9, color='black')
ax_precip.tick_params(axis='y', labelcolor='black')
ax_precip.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)
ax_precip.set_ylim(0, 17)
ax_precip.set_yticks([5, 10, 15])







# --- Section 3: Relative Humidity 2m (Wetterzentrale-style polished split) ---
ax_rh = axs[2]

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

rh = np.array(relative_humidity_2m, dtype=float)

n = (len(rh) // 3) * 3
rh_3h = rh[:n].reshape(-1, 3).mean(axis=1)
time_3h = time_arr[:n].reshape(-1, 3)[:, 0]

x = np.array(time_3h)
y = np.array(rh_3h)

ax_rh.set_ylim(0, 100)

# --- smooth interpolation ---
x_dense = np.linspace(x.min(), x.max(), 350)
y_dense = np.interp(x_dense, x, y)

Z = np.tile(np.linspace(0, 100, 220)[:, None], (1, len(x_dense)))
Y_curve = np.tile(y_dense, (220, 1))

# =========================
# NICE GREY (0–60%)
# =========================
grey_cmap = LinearSegmentedColormap.from_list(
    "grey_soft",
    [
        "#f5f5f5",
        "#e0e0e0",
        "#cfcfcf",
        "#b8b8b8",
        "#9e9e9e"
    ]
)

mask_grey = (Z <= Y_curve) & (Z <= 60)
Z_grey = np.ma.masked_where(~mask_grey, Z)

ax_rh.imshow(
    Z_grey,
    extent=[x.min(), x.max(), 0, 100],
    origin='lower',
    aspect='auto',
    cmap=grey_cmap,
    alpha=0.55,
    zorder=1
)

# =========================
# NICE GREEN (>60%)
# =========================
green_cmap = LinearSegmentedColormap.from_list(
    "green_soft",
    [
        "#e8f5e9",
        "#c8e6c9",
        "#a5d6a7",
        "#66bb6a",
        "#2e7d32"
    ]
)

mask_green = (Z <= Y_curve) & (Z > 60)
Z_green = np.ma.masked_where(~mask_green, Z)

ax_rh.imshow(
    Z_green,
    extent=[x.min(), x.max(), 0, 100],
    origin='lower',
    aspect='auto',
    cmap=green_cmap,
    alpha=0.65,
    zorder=2
)

# --- outline curve ---
ax_rh.plot(x, y, color='black', linewidth=0.8, zorder=3)

# --- styling ---
ax_rh.set_ylabel('RH\n(%)', fontsize=9)
ax_rh.set_yticks([20, 40, 60, 80, 100])

ax_rh.grid(axis='y', linestyle='dotted', alpha=0.5)

ax_rh.axhline(60, color='gray', linestyle='--', linewidth=1.2, alpha=0.9)

















# --- Section 4: Pressure ---
ax_pressure = axs[3]

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










# --- Section 5: Temperature 2m ---
ax_temp = axs[4]  
ax_temp.plot(times, temperature_2m, color='#FF8000', linewidth=1.5)
ax_temp.set_ylabel('T2m\n(°C)', fontsize=10, color='black')
ax_temp.tick_params(axis='y', labelcolor='black')
ax_temp.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# --- Y-axis limits and ticks ---
tmin_rounded = int(np.floor(temperature_2m.min() / 2.0) * 2)
tmax_actual = temperature_2m.max()
tmax_with_margin = int(np.ceil(tmax_actual)) + 5   # add +5 margin for annotation boxes

ax_temp.set_ylim(tmin_rounded, tmax_with_margin)

yticks = np.arange(tmin_rounded, tmax_with_margin + 1, 4)
ax_temp.set_yticks(yticks)

# --- Plot daily maxima between 06–18 UTC ---
import pandas as pd

times_index = pd.DatetimeIndex(times)
df_temp = pd.DataFrame({'temp': temperature_2m}, index=times_index)

grouped = df_temp.groupby(df_temp.index.date)

y_offset = 1.5  # offset above the line

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

# hide top y-tick label
yt = ax_temp.get_yticks()
ax_temp.set_yticks(yt)
ax_temp.set_yticklabels([str(int(v)) if i != len(yt)-1 else '' for i, v in enumerate(yt)])











# --- Section 6: Wind Gusts and Beaufort Scale Visualization ---
ax_windgust = axs[5]

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

















# Section 7: CAPE
ax_cape = axs[6]

ax_cape.set_ylabel('MUCAPE\n(J/kg)', fontsize=9, color='#007F7F')
ax_cape.tick_params(axis='y', labelcolor='#007F7F')

cape_line, = ax_cape.plot(
    times,
    cape,
    color='#007F7F',
    linewidth=1.5,
    label='MUCAPE (J/kg)',
    zorder=10
)

ax_cape.grid(axis='both', color='#92A9B6',
              linestyle='dotted', dashes=(2, 5),
              alpha=0.7, zorder=0)

max_cape = np.nanmax(cape)
ymax = max_cape + 200

step = 200 if max_cape < 1000 else 400

ax_cape.set_ylim(0, ymax)
ax_cape.set_yticks(np.arange(step, ymax + 1, step))

ax_cape.set_frame_on(True)
ax_cape.patch.set_visible(True)
ax_cape.patch.set_alpha(1)

for side in ['top', 'right', 'left', 'bottom']:
    ax_cape.spines[side].set_visible(True)
    ax_cape.spines[side].set_linewidth(1)

# Layout
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
        tick, -0.25,
        label,
        ha='center',
        va='top',
        transform=axs[-1].get_xaxis_transform(),
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
        label,
        ha='center',
        va='bottom',
        transform=axs[0].get_xaxis_transform(),
        fontsize=10
    )

day_labels = [mdates.num2date(t).strftime('%a').upper() for t in ticks_00z]

for tick, day in zip(ticks_00z, day_labels):
    axs[0].text(
        tick, 1.53,   
        day,
        ha='center',
        va='bottom',
        transform=axs[0].get_xaxis_transform(),
        fontsize=9,
        fontweight='bold',
        color='black'
    )


filename = f"kiato_ecmwf.png"
plt.subplots_adjust(hspace=0.05)
plt.savefig(filename, dpi=96, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)














