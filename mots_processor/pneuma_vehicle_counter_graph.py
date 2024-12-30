#type: ignore

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pneuma_vehicle_counter import *

def tse(df_t: pd.DataFrame, mot_file: str, u: int, bottom: bool, results: list, values: list):
    print(f"Evaluating {mot_file}")

    df = load_df(mot_file, bottom)
    speeds, flows, dens, counts = calc_tse(df, u, args.spacing)
    speeds_t, flows_t, dens_t, counts_t = calc_tse(df_t, u, args.spacing)
    print("speeds", np.array(speeds_t), np.array(speeds))
    print("counts", counts_t, counts)

    errors = {
        "Counts": [100 * abs(c - counts[i]) / c if c > 0 else 0 for i, c in enumerate(counts_t)],
        "Flows": [100 * abs(f - flows[i]) / f if f > 0 else 0 for i, f in enumerate(flows_t)],
        "Speeds": [100 * abs(s - speeds[i]) / s if s > 0 else 0 for i, s in enumerate(speeds_t)],
        "Densities": [100 * abs(d - dens[i]) / d if d > 0 else 0 for i, d in enumerate(dens_t)],
    }
    errors_abs = {
        "Counts": [abs(c - counts[i]) for i, c in enumerate(counts_t)],
        "Flows": [abs(f - flows[i]) for i, f in enumerate(flows_t)],
        "Speeds": [abs(s - speeds[i]) for i, s in enumerate(speeds_t)],
        "Densities": [abs(d - dens[i]) for i, d in enumerate(dens_t)],
    }
    values_p = {
        "Counts": counts,
        "Flows": flows,
        "Speeds": speeds,
        "Densities": dens,
    }
    values_t = {
        "Counts": counts_t,
        "Flows": flows_t,
        "Speeds": speeds_t,
        "Densities": dens_t,
    }
    values.append({"true": values_t, "pred": values_p})

    for name, errs in errors.items():
        results.append(errs)
    for name, errs in errors_abs.items():
        results.append(errs)


def plot_bars(data, column, area, u):
    width = 0.25  # Bar width
    group_spacing = 1
    x = np.arange(len(data[0])) * (len(trackers) * width + group_spacing)

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, tracker_data in enumerate(data):
        ax.bar(x + i * width, tracker_data, width, label=trackers[i])

    ax.set_ylabel(f"{column}")
    ax.set_xlabel('Data Points')
    ax.set_title(f'{column} for Area {area}, U {u}')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'P {i+1}' for i in range(len(data[0]))])
    ax.legend(title='Trackers')

    plt.tight_layout()

    img_name = column.replace(" ", "-").replace("Absolute", "Cummulative")
    img_title = f"raw-{img_name}-{area}-{u}"
    plt.savefig(f'imgs/{area}/{img_title}.png')
    plt.clf()

def show_bar(trackers, res_df, area, u):
    for column in res_df.columns:
        data = [res_df[column][f'{tracker}-7'] for tracker in trackers]
        plot_bars(data, column, area, u)

def show_plot(trackers, values, area, u):
    names = ["Counts", "Flows", "Speeds", "Densities"]
    plt.rcParams["figure.figsize"] = (16,9)
    for name in names:
        for tracker in trackers:
            for vs in values[f"{tracker}-7"]:
                tracker_name = tracker.upper().replace('-', ' ')
                plt.plot(vs["pred"][name], 'o--', label = f"{tracker_name} Predicted values")

        plt.title(f"Predicted and ground truth values for {name}")
        if name == names[0]:
            plt.ylabel("Vehicles")
        elif name == names[1]:
            plt.ylabel("Vehicles / hour")
        elif name == names[2]:
            plt.ylabel("Km / hour")
        elif name == names[3]:
            plt.ylabel("Vehicles / Km")

        plt.plot(vs["true"][name], 'o-', label = "True values")
        plt.xlabel("Data")
        plt.legend()
        plt.tight_layout()

        img_title = f"{name}-{area}-{u}"
        plt.savefig(f'imgs/{area}/{img_title}.png')
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index')
    parser.add_argument('-s', '--spacing', type=int, default=100)
    parser.add_argument('-u', '--u', type=int)
    parser.add_argument('-b', '--bottom', action='store_true')
    args = parser.parse_args()

    truth_mot = f"../{args.index}_0900_0930_D10_RM_mot.txt"
    df_t = load_df(truth_mot, args.bottom)

    results = {}
    values = {}

    column_names = []

    names = ["Counts", "Flows", "Speeds", "Densities"]
    for name in names:
        column_names.append(f"Relative error {name}")
    for name in names:
        column_names.append(f"Absolute error {name}")

    for kind in ["vel", "acc", "adp"]:
        for i in range(7, 8):
            key = f"sort-{kind}-{i}"
            results[key] = []
            values[key] = []
            mot_file = f"../sort-acc/output/pNEUMA{args.index}-{i}-tiny-{kind}.mot"
            tse(df_t, mot_file, args.u, args.bottom, results[key], values[key])

    for tracker in ["idm-tracker", "ByteTrack", "deepsort"]:
        for i in range(7, 8):
            key = f"{tracker}-{i}"
            results[key] = []
            values[key] = []
            mot_file = f"../{tracker}/output/pNEUMA{args.index}-{i}-tiny.mot"
            tse(df_t, mot_file, args.u, args.bottom, results[key], values[key])

    res_df = pd.DataFrame(results).T
    res_df.columns = column_names

    trackers = ["sort-vel", "sort-acc", "sort-adp", "idm-tracker", "ByteTrack", "deepsort"]
    show_plot(trackers, values, args.index, args.u)
    show_bar(trackers, res_df, args.index, args.u)

