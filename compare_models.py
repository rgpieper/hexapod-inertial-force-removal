
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from format_data import load_perts_h5
from mlp_model import BasicMLP, calc_avg_vaf
from lstm_model import MIMOLSTM
import matplotlib.pyplot as plt
import IPython

if __name__ == "__main__":

    window_size = 500
    step_size = 50
    accel_chans = 12
    force_chans = 8
    train_ratio = 0.8

    input_dim = accel_chans*window_size
    output_dim = force_chans*window_size
    
    data_file = "data/noLoadPerts_131125.h5"

    accel_segs, force_segs, meta_data = load_perts_h5(data_file)

    # replication training split
    _, val_pairs = train_test_split(
        list(zip(accel_segs, force_segs, meta_data)),
        test_size=1.0-train_ratio,
        random_state=42 # seed for reproducibility
    )
    val_accel_segments, val_force_segments, val_meta_data = zip(*val_pairs)

    val_meta_df = pd.DataFrame(val_meta_data)
    print("-----VALIDATION SEGMENTS-----")
    default_rows = pd.get_option('display.max_rows')
    pd.set_option('display.max_rows', None)
    print(val_meta_df)
    pd.set_option('display.max_rows', default_rows)

    mlpmodel_file = "models/mlp_models_20251114_005049/mlp_general.pth"
    MLPModel = BasicMLP(
        input_dim=input_dim,
        output_dim=output_dim
    )
    MLPModel.load_weights(mlpmodel_file)
    predictions_mlp = MLPModel.predict_segments(
        inputs=val_accel_segments,
        step_size=step_size
    )

    lstmmodel_file = "models/lstm_models_###/lstm_general.pth"
    LSTMModel = MIMOLSTM(
        input_size=accel_chans,
        output_size=force_chans
    )
    LSTMModel.load_weights(lstmmodel_file)
    predictions_lstm = LSTMModel.predict_segments(
        inputs=val_accel_segments,
        window_size=window_size,
        step_size=step_size
    )

    # plot predictions
    seg_idx = 2
    IPython.embed()
    seg_slice = range(0,1000)

    measurement = val_force_segments[seg_idx][seg_slice,:]
    prediction_mlp = predictions_mlp[seg_idx][seg_slice,:]
    prediction_lstm = predictions_lstm[seg_idx][seg_slice,:]
    seg_meta_data = val_meta_data[seg_idx]
    seg_direction = "forward" if seg_meta_data["dir"] == 1 else "reverse"

    _, match_force_segs, match_meta_data = load_perts_h5(
        data_file,
        dirs=[seg_meta_data["dir"]],
        axes_x=[seg_meta_data["axis_x"]],
        axes_z=[seg_meta_data["axis_z"]]
    )
    other_force_segs = []
    for i, info in enumerate(match_meta_data):
        if info["name"] != seg_meta_data["name"]:
            other_force_segs.append(match_force_segs[i])
    avg_seg = np.mean(np.stack(other_force_segs), axis=0)[seg_slice,:]

    vaf_pred_mlp = calc_avg_vaf([torch.from_numpy(prediction_mlp).float()], [torch.from_numpy(measurement).float()])
    vaf_pred_lstm = calc_avg_vaf([torch.from_numpy(prediction_lstm).float()], [torch.from_numpy(measurement).float()])
    vaf_avg = calc_avg_vaf([torch.from_numpy(avg_seg).float()], [torch.from_numpy(measurement).float()])

    rows = 4
    cols = 2

    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)

    fig.suptitle(f"Direction: {seg_direction}, X Pos: {seg_meta_data["axis_x"]}, Z Pos: {seg_meta_data["axis_z"]}", fontweight='bold')

    for i in range(rows):

        for j in range(cols):

            chan_idx = i + j*4

            ax = axs[i,j]

            ax.plot(measurement[:,chan_idx], label="Measurement")
            ax.plot(avg_seg[:,chan_idx], label=f"Averaged: {vaf_avg:.01f}%", linestyle="--")
            ax.plot(prediction_lstm[:,chan_idx], label=f"LSTM Prediction: {vaf_pred_lstm:.01f}%")
            ax.plot(prediction_mlp[:,chan_idx], label=f"MLP Prediction: {vaf_pred_mlp:.01f}%")

            if i==0 and j==0:
                ax.legend()
    
    plt.tight_layout()

    plt.show()




