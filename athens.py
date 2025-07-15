import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import pandas as pd
import requests
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from datetime import datetime, timedelta, timezone
import time

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
latitude = 38
longitude = 24

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
        "winddirection_10m",
        "cape",
        "lifted_index",
        "precipitation",            
        "showers",         
        "snowfall",
        "geopotential_height_1000hPa",
        "geopotential_height_500hPa",
        "geopotential_height_100hPa",
        "freezing_level_height"
    ]),
    "forecast_days": 5,
    "timezone": "UTC",
    "models": "gfs_seamless"
}

max_retries = 5
delay_seconds = 10

for attempt in range(1, max_retries + 1):
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        break
    except (requests.exceptions.RequestException) as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt == max_retries:
            print("Max retries reached. Exiting.")
            raise
        else:
            print(f"Retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)

response = requests.get(url, params=params)
data = response.json()

first_forecast_time = pd.to_datetime(data['hourly']['time'][0])

if "generationtime_ms" in data:
    gfs_generation_time = latest_run_time
else:
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

# New variables for DAM section
geopotential_1000 = get_data("geopotential_height_1000hPa")
geopotential_500 = get_data("geopotential_height_500hPa")
geopotential_100 = get_data("geopotential_height_100hPa")
freezing_level = get_data("freezing_level_height") 
dam = (geopotential_500 - geopotential_1000) / 10  # convert to dam

time_nums = mdates.date2num(times)

fig, axs = plt.subplots(
    6, 1,
    figsize=(1000 / 96, 900 / 96),
    gridspec_kw={'height_ratios': [1, 2.5, 1.1, 1, 1, 1]},
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

ax_cloud.set_title(f"ATHENS Init: {latest_run_time:%Y-%m-%d} {latest_run_time:%HZ}",
                   loc="center", fontsize=14, fontweight='bold', color='black', y=1.5)

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
    ["" if p == 1000 else str(p) for p in pressure_levels],
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
        linewidth=0.5
    )

# Add humidity contour lines
contour_lines = ax_humidity.contour(
    T, P, humidity,
    levels=bounds[1:],  # Skip first level to avoid lightest shade
    colors='black',
    linewidths=0.8,
    linestyles='--'
)

# Label contours
ax_humidity.clabel(
    contour_lines,
    fmt='%d',
    fontsize=8,
    inline=True
)



# Section 3: Precipitation
ax_precip = axs[2]
bar_width = (time_nums[1] - time_nums[0]) * 1.8
bar_width_showers = (time_nums[1] - time_nums[0]) * 0.9

time_nums_3h = time_nums[::3]
rain_3h = precipitation[::3]
showers_3h = showers[::3]
snowfall_3h = snowfall[::3]

ax_precip.bar(time_nums_3h, rain_3h, width=bar_width, color='#20D020', alpha=1.0, label='Rain')
ax_precip.bar(time_nums_3h, showers_3h, width=bar_width_showers, color='#FA3C3C', alpha=1.0, label='Showers')
ax_precip.bar(time_nums_3h, snowfall_3h, width=bar_width, color='#4040FF', alpha=1.0, label='Snowfall')
ax_precip.set_ylabel('Precipitation\n(mm)', fontsize=9, color='black')
ax_precip.tick_params(axis='y', labelcolor='black')
ax_precip.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# Set Y-axis dynamically
max_precip = max(np.max(rain_3h), np.max(showers_3h), np.max(snowfall_3h))
if max_precip <= 10:
    y_step = 2
elif max_precip <= 30:
    y_step = 5
else:
    y_step = 10

y_max = np.ceil(max_precip + 2)  # add 2 mm margin
y_max = y_step * np.ceil(y_max / y_step)  # round up to nearest step

ax_precip.set_ylim(0, y_max)
ax_precip.set_yticks(np.arange(0, y_max + y_step, y_step)[1:])


# Section 4: Pressure & 10m Winds
ax_pressure = axs[3]
ax_pressure.plot(times, pressure_msl, color='#00A0FF', linewidth=1, label='SLP (hPa)')
ax_pressure.set_ylabel('SLP\n(hPa)', fontsize=9, color='black')
ax_pressure.tick_params(axis='y', labelcolor='black')
ax_pressure.grid(axis='x', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# Twin axis for windspeed
ax_wind = ax_pressure.twinx()
wind_knots = windspeed_10m * 0.539957
ax_wind.plot(times, wind_knots, color='green', linestyle='--', linewidth=1.8)
ax_wind.plot(times[::3], wind_knots[::3], marker='o', linestyle='None',
             markerfacecolor='none', markeredgecolor='green', markersize=6)
ax_wind.set_ylabel('Wind\n(knots)', fontsize=9, color='black')
ax_wind.tick_params(axis='y', labelcolor='black')
ax_wind.grid(axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)

# --- FIX: Plot wind barbs inside ax_pressure box ---
u10 = -wind_knots * np.sin(np.deg2rad(winddirection_10m))
v10 = -wind_knots * np.cos(np.deg2rad(winddirection_10m))

# Choose a fixed Y position inside ax_pressure for all barbs
barb_y = pressure_msl.min() + 5  # slightly above the min pressure
ax_pressure.barbs(time_nums[::3], [barb_y] * len(time_nums[::3]), u10[::3], v10[::3],
                  length=6, linewidth=0.5, color='black', zorder=3)

# Set y-axis limits and ticks for pressure
p_min_rounded = int(np.floor(pressure_msl.min() / 5.0) * 5)
p_max_rounded = int(np.ceil(pressure_msl.max() / 5.0) * 5)
ax_pressure.set_ylim(p_min_rounded, p_max_rounded)
ax_pressure.yaxis.set_major_locator(ticker.MultipleLocator(5))
pressure_ticks = np.arange(p_min_rounded, p_max_rounded + 1, 5)
ax_pressure.set_yticks(pressure_ticks[1:])  # exclude first tick

# Set y-axis limits and ticks for windspeed
wind_min = max(0, wind_knots.min() - 5)
wind_max = wind_knots.max() + 5
wind_min_floor = np.floor(wind_min)
wind_max_ceil = np.ceil(wind_max)
ax_wind.set_ylim(wind_min_floor, wind_max_ceil)
wind_ticks = np.arange(wind_min_floor, wind_max_ceil + 1, 5)
ax_wind.set_yticks(wind_ticks[1:])  # exclude first tick







# Section 5: CAPE and Lifted Index
ax_cape = axs[4]
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
cape_line, = ax_cape.plot(times, cape, color='black', label='CAPE (J/kg)', zorder=10)

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
ax_li.set_ylabel('Lifted index', fontsize=9, color='#F08228')
ax_li.tick_params(axis='y', labelcolor='black')
ax_li.set_ylim(0, -6)
ax_li.set_yticks(np.arange(-2, -8, -2))






# Section 6: DAM and Freezing Level
ax_dam = axs[5]
ax_dam.plot(times, dam, color='red', linewidth=1.5, label='DAM (500–1000 hPa)')

# Set DAM y-axis limits and ticks
dam_min = min(dam)
dam_max = max(dam)

# Round limits to nearest multiple of 5
dam_lower = np.floor((dam_min - 5) / 5) * 5
dam_upper = np.ceil((dam_max + 5) / 5) * 5

# Set y-limits and yticks
ax_dam.set_ylim(dam_lower, dam_upper)
ax_dam.set_yticks(np.arange(dam_lower, dam_upper + 1, 5))

# Label and style
ax_dam.set_ylabel("Z500-Z1000\n(dm)", fontsize=9, color='red')
ax_dam.tick_params(axis='y', labelcolor='red')
ax_dam.grid(which='both', axis='both', color='#92A9B6', linestyle='dotted', dashes=(2, 5), alpha=0.8)


# Add freezing level on secondary y-axis
ax_dam2 = ax_dam.twinx()
ax_dam2.plot(times, freezing_level, color='blue', linestyle='--', linewidth=1.5, label='Freezing Level')

# Set Freezing Level y-axis limits and ticks
freeze_min = min(freezing_level)
freeze_max = max(freezing_level)

# Expand limits by ±100
freeze_lower = np.floor((freeze_min - 100) / 100) * 100
freeze_upper = np.ceil((freeze_max + 100) / 100) * 100

# Determine step size based on range
range_size = freeze_upper - freeze_lower
if range_size <= 500:
    step = 100
elif range_size <= 1000:
    step = 200
else:
    step = 400

# Set y-limits
ax_dam2.set_ylim(freeze_lower, freeze_upper)

# Set yticks
yticks_freeze = np.arange(freeze_lower, freeze_upper + step, step)
ax_dam2.set_yticks(yticks_freeze)

# Label and style
ax_dam2.set_ylabel("Freezing level\n (m)", fontsize=9, color='blue')
ax_dam2.tick_params(axis='y', labelcolor='blue')

# Optional legend
# lines1, labels1 = ax_dam.get_legend_handles_labels()
# lines2, labels2 = ax_dam2.get_legend_handles_labels()
# ax_dam2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)








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
        tick, -0.25,
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
        tick, 1.30,
        label, ha='center', va='bottom',
        transform=axs[0].get_xaxis_transform(which='grid'),
        fontsize=10
    )

plt.subplots_adjust(hspace=0.05)
plt.savefig("athens.png", dpi=96, bbox_inches='tight', pad_inches=0)
plt.close(fig)
