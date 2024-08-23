import argparse
import pandas as pd

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mot-file')
    args = parser.parse_args()
    df = load_df(args.mot_file)
    print(len(df["id"].unique()))
    passed_through_pos(df, 500)
