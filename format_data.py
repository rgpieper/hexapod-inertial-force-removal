#!/usr/bin/python

import ezc3d
import numpy as np
import numpy.typing as npt
from typing import List, Optional
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
import IPython
import matplotlib.pyplot as plt

# NOTE: Nexus graph depicts first analog datapoint to align with frame 1 and datapoint 5 aligning with frame 2

class C3DMan:
    def __init__(self, c3d_path: str):

        self.c3d_path = c3d_path
        self.c3d_data = ezc3d.c3d(self.c3d_path)

        desc_analog = self.c3d_data["parameters"]["ANALOG"]["DESCRIPTIONS"]["value"]
        lab_analog = self.c3d_data["parameters"]["ANALOG"]["LABELS"]["value"]
        self.analogs_idx = {det: i for i, det in enumerate(list(zip(desc_analog,lab_analog)))}
        self.fs_analog = self.c3d_data["header"]["analogs"]["frame_rate"]

        desc_point = self.c3d_data["parameters"]["POINT"]["DESCRIPTIONS"]["value"]
        lab_point = self.c3d_data["parameters"]["POINT"]["LABELS"]["value"]
        self.points_idx = {det: i for i, det in enumerate(list(zip(desc_point,lab_point)))}
        self.fs_points = self.c3d_data["header"]["points"]["frame_rate"]

        self.accel_df = pd.DataFrame()
        self.rawforce_df = pd.DataFrame()
        self.fs = self.fs_analog # unless otherwise modified

    def extract_accel(
            self,
            desc_labs: dict[str,tuple[str,str]] = {
                'a1x': ('Analog Accelerometer::Acceleration [19,1]', 'Acceleration.X'),
                'a1y': ('Analog Accelerometer::Acceleration [19,2]', 'Acceleration.Y'),
                'a1z': ('Analog Accelerometer::Acceleration [19,3]', 'Acceleration.Z'),
                'a2x': ('Analog Accelerometer::Acceleration [20,1]', 'Acceleration.X'),
                'a2y': ('Analog Accelerometer::Acceleration [20,2]', 'Acceleration.Y'),
                'a2z': ('Analog Accelerometer::Acceleration [20,3]', 'Acceleration.Z'),
                'a3x': ('Analog Accelerometer::Acceleration [21,1]', 'Acceleration.X'),
                'a3y': ('Analog Accelerometer::Acceleration [21,2]', 'Acceleration.Y'),
                'a3z': ('Analog Accelerometer::Acceleration [21,3]', 'Acceleration.Z'),
                'a4x': ('Analog Accelerometer::Acceleration [22,1]', 'Acceleration.X'),
                'a4y': ('Analog Accelerometer::Acceleration [22,2]', 'Acceleration.Y'),
                'a4z': ('Analog Accelerometer::Acceleration [22,3]', 'Acceleration.Z')
            }
    ):
        
        accel_idx = [self.analogs_idx[q] for q in desc_labs.values()]
        accel_data = self.c3d_data["data"]["analogs"][0][accel_idx]

        self.accel_df[list(desc_labs.keys())] = accel_data.T

    def extract_rawforce(
            self,
            desc_labs: dict[str,tuple[str,str]] = {
                'fx12': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FX12'),
                'fx34': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FX34'),
                'fy14': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FY14'),
                'fy23': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FY23'),
                'fz1': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FZ1'),
                'fz2': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FZ2'),
                'fz3': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FZ3'),
                'fz4': ('Kistler Force Plate 2.0.0.0::Raw [27]', 'Raw.FZ4')
            }
    ):
        
        rawforce_idx = [self.analogs_idx[q] for q in desc_labs.values()]
        rawforce_data = self.c3d_data["data"]["analogs"][0][rawforce_idx]

        self.rawforce_df[list(desc_labs.keys())] = rawforce_data.T

    def segment_trainperts(
            self,
            t_segment: float,
            threshold: float,
            t_base: float = 0.1,
            t_buffer: float = 0.025
    ):
        
        n_segment = math.ceil(t_segment*self.fs)
        n_base = math.ceil(t_base*self.fs)
        n_buffer = math.ceil(t_buffer*self.fs)
        
        a_norms = pd.DataFrame()
        
        a_norms["a1n"] = np.linalg.norm(self.accel_df[["a1x", "a1y", "a1z"]].values, axis=1) # accelerometer 1
        a_norms["a2n"] = np.linalg.norm(self.accel_df[["a2x", "a2y", "a2z"]].values, axis=1) # accelerometer 2
        a_norms["a3n"] = np.linalg.norm(self.accel_df[["a3x", "a3y", "a3z"]].values, axis=1) # accelerometer 3
        a_norms["a4n"] = np.linalg.norm(self.accel_df[["a4x", "a4y", "a4z"]].values, axis=1) # accelerometer 4

        a_base = a_norms.iloc[0:n_base].mean()

        accel_segments = []
        rawforce_segments = []
        i = n_base
        while i < len(a_norms):
            if any(abs(a_norms.iloc[i] - a_base) > threshold):
                i_start = i-n_buffer if i-n_buffer >= 0 else 0
                i_next = i_start+n_segment if i_start+n_segment < len(a_norms) else len(a_norms)
                accel_seg = self.accel_df.iloc[i_start:i_next].values
                accel_segments.append(accel_seg)
                rawforce_seg = self.rawforce_df.iloc[i_start:i_next].values
                rawforce_segments.append(rawforce_seg)
                i = i_next
            else:
                i = i + 1

        return accel_segments, rawforce_segments

class FlattenedWindows(Dataset):
    def __init__(
            self,
            input_arrays: List[npt.ArrayLike],
            window_size: int,
            step_size: int,
            target_arrays: Optional[List[npt.ArrayLike]] = None
    ):
        
        self.window_size = window_size
        self.step_size = step_size

        self.input_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in input_arrays]
        if target_arrays is not None:
            self.target_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in target_arrays]

        self.index_map = []

        for series_idx, input_series in enumerate(self.input_tensors):
            series_length = input_series.shape[0]
            max_start = series_length - self.window_size
            if max_start >= 0:
                start_indices = np.arange(0, series_length-self.window_size+self.step_size, self.step_size)
                valid_start_indices = start_indices[start_indices <= max_start]
                series_indices = series_idx*np.ones_like(valid_start_indices)
                self.index_map.extend(list(zip(series_indices, valid_start_indices)))

        self.num_windows = len(self.index_map)

    def __len__(self):

        return self.num_windows
    
    def __getitem__(self, idx):

        series_idx, start_idx = self.index_map[idx]
        end_idx = start_idx + self.window_size

        input_series = self.input_tensors[series_idx] # shape: (window_size, n_input_channels)
        X_window = input_series[start_idx:end_idx,:].flatten() # shape: (n_input_channels*window_size,)
        if target_series is not None:
            target_series = self.target_tensors[series_idx] # shape: (window_size, n_output_channels)
            Y_window = target_series[start_idx:end_idx,:].flatten() # shape: (n_output_channels*window_size,)
            return X_window, Y_window
        else:
            return X_window

if __name__ == "__main__":

    TestTrial = C3DMan("data/fullgrid_unloaded_02.c3d")
    TestTrial.extract_accel()
    TestTrial.extract_rawforce()

    # TestTrial.accel_df.to_csv("data/accelerations_unloaded_02.csv", index=False)
    # TestTrial.rawforce_df.to_csv("data/forces_unloaded_02.csv", index=False)

    IPython.embed()