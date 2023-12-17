"""
Returns the paths contained in the given split image id.
ids range from 0 to 15 (image was split into 16 different patches)
"""

import sys
import pandas as pd 

id = int(sys.argv[1])

df = pd.read_csv("0900_0930_D10_RM_mot.txt",
                 names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])

tile_size_x = 4096 // 4
tile_size_y = 2100 // 4

print(f"tile size: ({tile_size_x}, {tile_size_y})")

init_x = tile_size_x * (id % 4)
init_y = tile_size_y * (id // 4)

print(f"inits: ({init_x}, {init_y})")

fdf = df[
    (df['bb_left'] > init_x) & (df['bb_left'] < init_x + tile_size_x) &
    (df['bb_top'] > init_y) & (df['bb_top'] < init_y + tile_size_y)
]

fdf['bb_top'] -= init_y
fdf['bb_left'] -= init_x


fdf.to_csv("10_0900_0930_D10_RM_mot.txt",
           index=False, header=False)

print(fdf.shape)
print(fdf)

