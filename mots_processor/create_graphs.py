import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

filepath = sys.argv[1]
df_full = pd.read_csv(filepath)

columns_groups = [
    ("Relative error mean Counts", "Relative error std Counts"),
    ("Relative error mean Flows", "Relative error std Flows"),
    ("Relative error mean Speeds", "Relative error std Speeds"),
    ("Relative error mean Densities", "Relative error std Densities"),
    ("Absolute error mean Counts", "Absolute error std Counts"),
    ("Absolute error mean Flows", "Absolute error std Flows"),
    ("Absolute error mean Speeds", "Absolute error std Speeds"),
    ("Absolute error mean Densities", "Absolute error std Densities"),
]

trackers = ["sort-vel", "sort-acc", "sort-adp", "idm-tracker", "ByteTrack", "deepsort"]

def combine_means_and_stds(means, stds):
    combined_mean = np.mean(means)
    combined_std = np.sqrt(
        np.mean(stds**2 + (means - combined_mean) ** 2)
    )
    return combined_mean, combined_std

results = {}

for tracker in trackers:
    for mean_col, std_col in columns_groups:
        df = df_full[df_full.iloc[:, 0].str.startswith(tracker)]
        means = df[mean_col].to_numpy() #type: ignore
        stds = df[std_col].to_numpy() #type: ignore

        combined_mean, combined_std = combine_means_and_stds(means, stds)

        results[tracker + " " + mean_col] = combined_mean
        results[tracker + " " + std_col] = combined_std

results_df = pd.DataFrame([results])
print(results_df)

for mean_col, std_col in columns_groups:
    means = results_df[[c for c in results_df.columns if c.endswith(mean_col)]].values[0]
    stds = results_df[[c for c in results_df.columns if c.endswith(std_col)]].values[0]

    tse_kind = mean_col.split(" ")[-1]
    error_kind = mean_col.split(" ")[0]

    x = np.arange(len(trackers))  # The x locations for the groups
    width = 0.6  # Bar width

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        x,
        means,
        width,
        yerr=stds,
        color=['skyblue', 'orange', 'green', 'yellow', 'magenta', 'red'],
    )

    # Add labels, title, and custom ticks
    ax.set_ylabel('Relative Error (Mean ± Std)')
    ax.set_xlabel('Method')
    ax.set_title(f'Comparison of {error_kind} Errors for {tse_kind} Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(trackers)

    # Add mean values on top of bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005, f'{mean:.2f}', ha='center', va='bottom')

    # Show plot
    plt.tight_layout()
    plt.show()

