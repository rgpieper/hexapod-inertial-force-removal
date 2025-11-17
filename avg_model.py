
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from format_data import load_perts_h5
from mlp_model import calc_avg_vaf

if __name__ == "__main__":

    train_ratio = 0.8

    data_file = "data/noLoadPerts_131125.h5"

    model_info_list = []

    ### PERFORMANCE ON GENERAL AVERAGE

    accel_segs, force_segs, meta_data = load_perts_h5(data_file)

    # split segments for training / validation
    train_pairs, val_pairs = train_test_split(
        list(zip(accel_segs, force_segs)),
        test_size=1.0-train_ratio,
        random_state=42 # seed for reproducibility
    )
    _, train_force_segments = zip(*train_pairs)
    _, val_force_segments = zip(*val_pairs)

    avg_seg = np.mean(np.stack(train_force_segments), axis=0)
    avg_segs = [avg_seg.copy() for _ in range(len(val_force_segments))]

    avg_vaf = calc_avg_vaf(
        [torch.from_numpy(seg).float() for seg in avg_segs],
        [torch.from_numpy(seg).float() for seg in val_force_segments]
    )

    model_info_list.append({
        "axis_x": "all",
        "axis_z": "all",
        "dir": "both",
        "vaf": avg_vaf
    })

    ### PERFORMANCE ON SPECIFIC AVERAGE

    axes_x = [0, 50, 100, 150, 200]
    axes_z = [63, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 96, 100]
    dirs = [-1, 1]

    for x in axes_x:

        for z in axes_z:

            for dir in dirs:

                accel_segs, force_segs, meta_data = load_perts_h5(data_file, [dir], [x], [z])

                # split segments for training / validation
                train_pairs, val_pairs = train_test_split(
                    list(zip(accel_segs, force_segs)),
                    test_size=1.0-train_ratio,
                    random_state=42 # seed for reproducibility
                )
                _, train_force_segments = zip(*train_pairs)
                _, val_force_segments = zip(*val_pairs)

                avg_seg = np.mean(np.stack(train_force_segments), axis=0)
                avg_segs = [avg_seg.copy() for _ in range(len(val_force_segments))]

                avg_vaf = calc_avg_vaf(
                    [torch.from_numpy(seg).float() for seg in avg_segs],
                    [torch.from_numpy(seg).float() for seg in val_force_segments]
                )

                model_info_list.append({
                    "axis_x": x,
                    "axis_z": z,
                    "dir": dir,
                    "vaf": avg_vaf
                })

    model_info_df = pd.DataFrame(model_info_list)
    model_info_df.to_csv("models/avg_model_info.csv", index=False)
    # avg individual VAF: 94.5%