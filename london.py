import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import pandas as pd
import requests
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import os

# Location
latitude = 51.54
longitude = -0.17

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
        "precipitation"
    ]),
    "forecast_days": 5,
    "timezone": "UTC",
    "models": "gfs_global"
}

response = requests.get(url, params=params)
data = response.json()

# Forecast metadata
first_forecast_time = pd.to_datetime(data['hourly']['time'][0])
run_hour = first_forecast_time.hour
run_date = first_forecast_time.date()
print(f"Using GFS run: {run_date} {run_hour:02d}Z")

# === Section 1: Clouds ===
times_cloud = pd.to_datetime(data['hourly']['time'])
cloud_low = np.array(data['hourly']['cloud_cover_low'])
cloud_mid = np.array(data['hourly']['cloud_cover_mid'])
cloud_high = np.array(data['hourly']['cloud_cover_high'])

time_nums_cloud = mdates.date2num(times_cloud)
dt_cloud = time_nums_cloud[1] - time_nums_cloud[0] if len(time_nums_cloud) > 1 else 1

# === Sections 2-5: Full data ===
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
time_nums = mdates.date2num(times)

# --- Plotting ---
fig, axs = plt.subplots(
    5, 1,
    figsize=(1000 / 96, 780 / 96),
    gridspec_kw={'height_ratios': [0.8, 2, 1, 1, 1]},
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

ax_cloud.set_title(f"LONDON Init: {run_date} {run_hour:02d}Z", fontsize=14, fontweight='bold', color='black', y=1.5)
ax_cloud.tick_params(axis='y', colors='black')
ax_cloud.set_xlim(time_nums_cloud[0], time_nums_cloud[-1])
ax_cloud.grid(axis='x', color='gray', linestyle='--', alpha=0.7)

# Section 2: Humidity & Winds
ax_humidity = axs[1]
T, P = np.meshgrid(time_nums, pressure_levels)
colors = ['#FFFFFF', '#EAF5EA', '#C8D7C8', '#78D778', '#00FF00', '#00BE00']
bounds = [0, 30, 50, 70, 90, 95]
cmap_rh = mcolors.ListedColormap(colors)
norm_rh = mcolors.BoundaryNorm(bounds, cmap_rh.N, extend='max')
cf = ax_humidity.contourf(T, P, humidity, levels=bounds, cmap=cmap_rh, norm=norm_rh, extend='max')
ax_humidity.set_ylim(1000, 650)
ax_humidity.set_yticks(pressure_levels)
ax_humidity.set_yticklabels(
    ["" if p == 1000 else str(p) for p in pressure_levels],
    fontsize=9
)
ax_humidity.grid(axis='x', color='gray', linestyle='--', alpha=0.7)
indices_3h = np.arange(0, len(times), 3)
for i, p in enumerate(pressure_levels):
    ws_knots = windspeed[i][indices_3h] * 0.539957
    theta = np.deg2rad(winddirection[i][indices_3h])
    u = -ws_knots * np.sin(theta)
    v = -ws_knots * np.cos(theta)
    ax_humidity.barbs(time_nums[indices_3h], np.full(len(indices_3h), p), u, v, length=6, linewidth=0.5)


contour_lines = ax_humidity.contour(
    T, P, humidity, levels=bounds[1:], colors='black', linewidths=0.8, linestyles='--'
)
ax_humidity.clabel(contour_lines, fmt='%d', fontsize=8, inline=True)


# Section 3: Precipitation
ax_precip = axs[2]
bar_width = (time_nums[1] - time_nums[0]) * 0.8
ax_precip.bar(time_nums, precipitation, width=bar_width, color='darkblue', alpha=1.0)
ax_precip.set_ylabel('Precipitation\n(mm)', fontsize=9, color='black')
ax_precip.tick_params(axis='y', labelcolor='black')
ax_precip.grid(axis='both', color='gray', linestyle='--', alpha=0.7)
ax_precip.set_ylim(0, 17)
y_ticks = np.arange(0, 18, 5)
ax_precip.set_yticks(y_ticks[1:])  # skip 0

# Section 4: Pressure & 10m Winds
ax_pressure = axs[3]
ax_pressure.plot(times, pressure_msl, color='#00A0FF', linewidth=1, label='SLP (hPa)')
ax_pressure.set_ylabel('SLP\n(hPa)', fontsize=9, color='black')
ax_pressure.tick_params(axis='y', labelcolor='black')
ax_pressure.grid(axis='x', color='gray', linestyle='--', alpha=0.7)

ax_wind = ax_pressure.twinx()
wind_knots = windspeed_10m * 0.539957
ax_wind.plot(times, wind_knots, color='green', linestyle='--', linewidth=1.8)
ax_wind.plot(times[::3], wind_knots[::3], marker='o', linestyle='None',
             markerfacecolor='none', markeredgecolor='green', markersize=6)
ax_wind.set_ylabel('Wind\n(knots)', fontsize=9, color='black')
ax_wind.tick_params(axis='y', labelcolor='black')
ax_wind.grid(axis='both', color='gray', linestyle='--', alpha=0.7)

barb_y = pressure_msl.min() - 10
u10 = -wind_knots * np.sin(np.deg2rad(winddirection_10m))
v10 = -wind_knots * np.cos(np.deg2rad(winddirection_10m))
ax_wind.barbs(time_nums[::3], wind_knots[::3], u10[::3], v10[::3], length=6, linewidth=0.5, color='black')

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
ax_cape.plot(times, cape, color='red', label='CAPE (J/kg)')
ax_cape.set_ylabel('CAPE\n(J/kg)', fontsize=9, color='black')
ax_cape.tick_params(axis='y', labelcolor='black')

li_times = times[::3]                # every 3 hours
li_values = lifted_index[::3]        # every 3rd value
colors = ['#E6DC32' if val >= 0 else '#F08228' for val in li_values]

# Plot Lifted Index as bars
ax_li = ax_cape.twinx()
ax_li.bar(li_times, li_values, color=colors, width=0.08, align='center')


ax_li.set_ylabel('Lifted index', fontsize=9, color='black')
ax_li.tick_params(axis='y', labelcolor='black')
ax_li.set_ylim(4, -4)
ax_li.set_yticks(np.arange(4, -5, -2))
ax_li.axhline(0, color='gray', linestyle='--', linewidth=0.8)

ax_cape.grid(axis='x', color='gray', linestyle='--', alpha=0.7)
ax_cape.set_ylim(0, 800)
ax_cape.set_yticks(np.arange(0, 1000, 200))











# X-axis formatting

# --- Adjust figure margins to avoid clipping ---
fig.subplots_adjust(top=1.00, bottom=0.15)  # Add more space top & bottom

# --- BOTTOM AXIS ---
# Major ticks: 00Z and 12Z
axs[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
axs[-1].xaxis.set_major_formatter(FuncFormatter(
    lambda x, pos: mdates.num2date(x).strftime('%HZ')
))
axs[-1].tick_params(axis='x', which='major', labelsize=9, pad=5)

# Add dates (e.g., 19JUL) BELOW the times at 00Z only
ticks_00z = [t for t in mdates.date2num(times) if mdates.num2date(t).hour == 0]
labels_00z = [mdates.num2date(t).strftime('%d%b').upper() for t in ticks_00z]

for tick, label in zip(ticks_00z, labels_00z):
    axs[-1].text(
        tick, -0.25,  # push dates further down
        label, ha='center', va='top',
        transform=axs[-1].get_xaxis_transform(which='grid'),
        fontsize=10
    )

# --- TOP AXIS ---
# Add secondary X-axis for 00Z/12Z ticks
ax_cloud_secondary_x = ax_cloud.secondary_xaxis('top')
ax_cloud_secondary_x.set_xticks(
    [t for t in mdates.date2num(times) if mdates.num2date(t).hour in [0, 12]]
)
ax_cloud_secondary_x.set_xticklabels(
    [mdates.num2date(t).strftime('%HZ') for t in mdates.date2num(times)
     if mdates.num2date(t).hour in [0, 12]],
    fontsize=9
)
ax_cloud_secondary_x.tick_params(axis='x', which='major', pad=5)

# Add dates (e.g., 19JUL) ABOVE the times at 00Z only
for tick, label in zip(ticks_00z, labels_00z):
    ax_cloud.text(
        tick, 1.30,  # push dates further up
        label, ha='center', va='bottom',
        transform=ax_cloud.get_xaxis_transform(which='grid'),
        fontsize=10
    )



plt.subplots_adjust(hspace=0.05)
plt.savefig("london_meteogram.png", dpi=96, bbox_inches='tight', pad_inches=0)
plt.close(fig)









# === Upload the generated image to your web server ===
from ftplib import FTP

# FTP credentials (replace with your real details)
FTP_HOST = os.environ.get("FTP_HOST")         # e.g. ftp.example.com
FTP_USER = os.environ.get("FTP_USER")         # e.g. parognosis_user
FTP_PASS = os.environ.get("FTP_PASS")         # e.g. secretpassword
FTP_FOLDER = os.environ.get("FTP_FOLDER")     # folder path on your server

print("🌐 Connecting to FTP server...")
ftp = FTP(FTP_HOST)
ftp.login(FTP_USER, FTP_PASS)

# Change to the target directory
ftp.cwd(FTP_FOLDER)

# Open the file and upload it
with open("london_meteogram.png", "rb") as file:
    ftp.storbinary("STOR london_meteogram.png", file)

ftp.quit()




