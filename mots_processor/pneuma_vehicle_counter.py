#type: ignore

import argparse
import numpy as np
import pandas as pd

def load_df(mot_file: str, bottom: bool = False) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    if bottom:
        df = df[df["bb_top"] > 300]

    return df


def passed_through_pos(df: pd.DataFrame, x: int):
    last_frame = df["frame"].max()
    current_frame = 1
    count = 0
    while current_frame < last_frame:
        next_frame = current_frame + 1
        current_positions = df[df["frame"] == current_frame][["id", "bb_left"]]
        next_positions = df[df["frame"] == next_frame][["id","bb_left"]]
        for current_pos_index in current_positions.iterrows():
            current_pos = current_pos_index[1]
            last_pos = current_pos["bb_left"]
            if last_pos > x:
                continue

            next_pos = next_positions[next_positions["id"] == current_pos["id"]]
            if next_pos.empty:
                continue

            new_pos = next_pos["bb_left"].values[0]
            if new_pos > x:
                count += 1

        current_frame += 1
        print(f"Frame {current_frame} count is {count}")
    return count


FPS = 2.5 # video at 25FPS with images taken every 10th sample

def calc_speeds(df: pd.DataFrame) -> pd.DataFrame:
    # values in pixels per second
    df = df.sort_values("frame")
    df["speed"] = df["bb_left"].diff().fillna(0) / df["frame"].diff().fillna(1) * FPS
    return df

def calc_tse(df: pd.DataFrame, u: int, spacing: int):
    df = df.groupby("id")[df.columns.tolist()].apply(calc_speeds)
    current_index = 0
    frame_count = df["frame"].max()
    speed_sums = []
    counts = []
    while current_index < frame_count:
        speed_sum = 0
        count = 0
        current_frame = current_index
        last_frame = current_frame + spacing
        while current_frame < last_frame:
            current_positions = df[df["frame"] == current_frame][["id", "bb_left", "speed"]]
            df_reset = df.reset_index(drop=True)
            current_ids = current_positions["id"]
            filtered_df = df_reset[df_reset["id"].isin(current_ids) & (df_reset["frame"] > current_frame)]
            next_positions = filtered_df.loc[filtered_df.groupby("id")["frame"].idxmin(), ["id", "bb_left", "speed"]]
            for current_pos_index in current_positions.iterrows():
                current_pos = current_pos_index[1]
                last_pos = current_pos["bb_left"]
                # has already passed through u
                if last_pos > u:
                    continue


                # no new pos
                next_pos = next_positions[next_positions["id"] == current_pos["id"]]
                if next_pos.empty:
                    continue

                # passed through u
                new_pos = next_pos["bb_left"].values[0]
                if new_pos > u:
                    count += 1
                    speed_sum += current_pos["speed"]

            current_frame += 1

        current_index = last_frame
        speed_sums.append(speed_sum)
        counts.append(count)

    # speeds = [3.6 * s / c if c > 0 else 0 for s, c in zip(speed_sums, counts)]
    # leave everything in pixels
    speeds = [3.6 * s / c if c > 0 else 0 for s, c in zip(speed_sums, counts)]
    flows = [3600* c / (spacing * 1 / FPS) for c in counts]
    #dens = [1000 * f / s if s > 0 else 0 for f, s in zip(flows, speeds)]
    dens = [1000 * f / s if s > 0 else 0 for f, s in zip(flows, speeds)]
    return speeds, flows, dens, counts


def tse(df_t: pd.DataFrame, mot_file: str, u: int, bottom: bool, results: list):
    print(f"Evaluating {mot_file}")

    df = load_df(mot_file, bottom)
    speeds, flows, dens, counts = calc_tse(df, u, args.spacing)
    speeds_t, flows_t, dens_t, counts_t = calc_tse(df_t, u, args.spacing)

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
    errors_sum = {
        "Counts": 100 * abs(sum(counts_t) - sum(counts)) / sum(counts_t) if sum(counts_t) > 0 else 0,
        "Flows": 100 * abs(sum(flows) - sum(flows_t)) / sum(flows_t) if sum(counts_t) > 0 else 0,
        "Speeds": 100 * abs(sum(speeds) - sum(speeds_t)) / sum(speeds_t) if sum(counts_t) > 0 else 0,
        "Densities": 100 * abs(sum(dens_t) - sum(dens)) / sum(dens_t) if sum(counts_t) > 0 else 0,
    }

    for name, errs in errors.items():
        print(f"{name} relative error is {np.mean(errs)}, +- {np.std(errs)}")
        results.append(np.mean(errs))
        results.append(np.std(errs))
    print("")
    for name, errs in errors_abs.items():
        print(f"{name} cummulative error is {np.sum(errs)}, total sum error: {errors_sum[name]}")
        results.append(np.sum(errs))
        results.append(errors_sum[name])
    print("")
    for name, errs in errors_abs.items():
        print(f"{name} mean abs error is {np.mean(errs)}, +- {np.std(errs)}")
        results.append(np.mean(errs))
        results.append(np.std(errs))

    print("\n\n")


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


    column_names = []

    names = ["Counts", "Flows", "Speeds", "Densities"]
    for name in names:
        column_names.append(f"Relative error mean {name}")
        column_names.append(f"Relative error std {name}")
    for name in names:
        column_names.append(f"Cummulative error mean {name}")
        column_names.append(f"Cummulative error absolute {name}")
    for name in names:
        column_names.append(f"Absolute error mean {name}")
        column_names.append(f"Absolute error std {name}")

    for kind in ["vel", "acc", "adp"]:
        for i in range(7, 12):
            key = f"sort-{kind}-{i}"
            results[key] = []
            mot_file = f"../sort-acc/output/pNEUMA{args.index}-{i}-tiny-{kind}.mot"
            tse(df_t, mot_file, args.u, args.bottom, results[key])

    for tracker in ["idm-tracker", "ByteTrack", "deepsort"]:
        for i in range(7, 12):
            key = f"{tracker}-{i}"
            results[key] = []
            mot_file = f"../{tracker}/output/pNEUMA{args.index}-{i}-tiny.mot"
            tse(df_t, mot_file, args.u, args.bottom, results[key])

    res_df = pd.DataFrame(results).T
    res_df.columns = column_names
    res_df.to_csv(f"res-{args.index}-{args.u}.csv")


