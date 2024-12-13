#type: ignore

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_df(mot_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        mot_file,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    bottom_df = df[df["bb_top"] > 300]
    return bottom_df


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
    df["speed"] = df["bb_left"].diff().fillna(0) * FPS
    return df

def calc_tse(df: pd.DataFrame, u: int, spacing: int):
    df = df.groupby("id").apply(calc_speeds)
    current_index = 0
    frame_count = df["frame"].max()
    speed_sums = []
    counts = []
    while current_index < frame_count:
        speed_sum = 0
        count = 0
        current_frame = current_index + 1
        last_frame = current_frame + spacing
        while current_frame < last_frame:
            next_frame = current_frame + 1
            current_positions = df[df["frame"] == current_frame][["id", "bb_left", "speed"]]
            next_positions = df[df["frame"] == next_frame][["id","bb_left", "speed"]]
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

        current_index += spacing
        speed_sums.append(speed_sum)
        counts.append(count)

    # speeds = [3.6 * s / c if c > 0 else 0 for s, c in zip(speed_sums, counts)]
    # leave everything in pixels
    speeds = [3600 * s / c if c > 0 else 0 for s, c in zip(speed_sums, counts)]
    flows = [3600 * c / (spacing * 1 / FPS) for c in counts]
    #dens = [1000 * f / s if s > 0 else 0 for f, s in zip(flows, speeds)]
    dens = [f / s if s > 0 else 0 for f, s in zip(flows, speeds)]
    return speeds, flows, dens, counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mot-file')
    parser.add_argument('-t', '--truth-mot')
    parser.add_argument('-s', '--spacing', type=int)
    parser.add_argument('-f', '--filter-speed', action='store_true')
    args = parser.parse_args()
    df = load_df(args.mot_file)

    df_t = load_df(args.truth_mot)
    print("calculating preds")
    speeds, flows, dens, counts = calc_tse(df, 500, args.spacing)
    print("calculating true")
    speeds_t, flows_t, dens_t, counts_t = calc_tse(df_t, 500, args.spacing)

    print(sum(counts), sum(counts_t))

    plt.figure()
    plt.plot(counts, '-o', label="counts")
    plt.plot(counts_t, '-o', label="true")
    plt.xlabel("T")
    plt.ylabel("Num Vehicles")
    plt.title("Counts")
    plt.legend()
    plt.savefig('veh-count.png')
    plt.show(block=False)

    plt.figure()
    plt.plot(speeds, '-o', label="speeds")
    plt.plot(speeds_t, '-o', label="true")
    plt.xlabel("T")
    plt.ylabel("Pixels / minute")
    plt.legend()
    plt.title("Speeds")
    plt.savefig('speed.png')
    plt.show(block=False)

    plt.figure()
    plt.plot(flows, '-o', label="flows")
    plt.plot(flows_t, '-o', label="true")
    plt.xlabel("T")
    plt.ylabel("Vehicles / hour")
    plt.title("Flows")
    plt.legend()
    plt.savefig('flows.png')
    plt.show(block=False)

    plt.figure()
    plt.plot(dens, '-o', label="dens")
    plt.plot(dens_t, '-o', label="true")
    plt.xlabel("T")
    plt.ylabel("Vehicles / Pixels")
    plt.title("Dens")
    plt.legend()
    plt.savefig('dens.png')
    plt.show(block=False)

    ## show error
    times = list(range(len(counts)))
    errors = {
        "Counts": [100 * abs(c - counts[i]) / c if c > 0 else 0 for i, c in enumerate(counts_t)],
        "Flows": [100 * abs(f - flows[i]) / f if f > 0 else 0 for i, f in enumerate(flows_t)],
        "Speeds": [100 * abs(s - speeds[i]) / s if s > 0 else 0 for i, s in enumerate(speeds_t)],
        "Densities": [100 * abs(d - dens[i]) / d if d > 0 else 0 for i, d in enumerate(dens_t)],
    }
    for name, errs in errors.items():
        print(f"{name} error is {np.mean(errs)}")

    for att, value in errors.items():
        plt.figure()
        x = np.arange(len(errors["Counts"]))  # the label locations
        width = 0.5  # the width of the bars

        rects = plt.bar(x, value, width, label=att)
        plt.bar_label(rects, padding=3)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.ylabel('Errors (%)')
        plt.title(f'Relative errors - {att}')
        plt.xticks(x + width, times)
        plt.legend(loc='upper left')

        plt.savefig(f'{att.lower()}-error.png')
        plt.show(block=False)

    input()

