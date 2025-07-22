# --- Plotting ---
fig, axs = plt.subplots(
    8, 1,
    figsize=(1400 / 130, 1100 / 150), # or 1200/120 and 1000/120
    gridspec_kw={'height_ratios': [1.2, 3, 0.7, 0.7, 1.5, 1, 1, 1]},
    sharex=True
)
