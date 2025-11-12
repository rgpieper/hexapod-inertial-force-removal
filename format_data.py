#!/usr/bin/python

import ezc3d
import numpy as np
import numpy.typing as npt
from typing import Type, List, Tuple, Optional, Any, Dict
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import IPython
import matplotlib.pyplot as plt

# NOTE: Nexus graph depicts first analog datapoint to align with frame 1 and datapoint 5 aligning with frame 2

class C3DMan:
    def __init__(self, c3d_path: str):

        self.c3d_path = c3d_path
        self.c3d_data = ezc3d.c3d(self.c3d_path)

        desc_analog = self.c3d_data["parameters"]["ANALOG"]["DESCRIPTIONS"]["value"]
        lab_analog = self.c3d_data["parameters"]["ANALOG"]["LABELS"]["value"]
        self.analog_desclab_df = pd.DataFrame({'Description': desc_analog, 'Label': lab_analog})
        self.analog_idx = {det: i for i, det in enumerate(list(zip(desc_analog,lab_analog)))}
        self.fs_analog = self.c3d_data["header"]["analogs"]["frame_rate"]

        desc_point = self.c3d_data["parameters"]["POINT"]["DESCRIPTIONS"]["value"]
        lab_point = self.c3d_data["parameters"]["POINT"]["LABELS"]["value"]
        self.points_idx = {det: i for i, det in enumerate(list(zip(desc_point,lab_point)))}
        self.fs_points = self.c3d_data["header"]["points"]["frame_rate"]

        self.accel_df = pd.DataFrame()
        self.rawaccel_df = pd.DataFrame()
        self.rawforce_df = pd.DataFrame()
        self.hextrigger_df = pd.DataFrame()
        self.fs = self.fs_analog # unless otherwise modified

    def print_analog_desclabs(self) -> None:

        print()
        print("---------- ANALOG SIGNALS ----------")
        default_rows = pd.get_option('display.max_rows')
        pd.set_option('display.max_rows', None)
        print(self.analog_desclab_df)
        print()
        pd.set_option('display.max_rows', default_rows)

    def extract_accel(
            self,
            name_mapping: dict[str,tuple[str,str]] = {
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
    ) -> None:

        self.accel_df = self.extract_analogs(name_mapping)

    def extract_rawaccel(
            self,
            name_mapping: dict[str,tuple[str,str]] = {
                'a1x': ('Generic Analog::Electric Potential [33,1]', 'Electric Potential.X'),
                'a1y': ('Generic Analog::Electric Potential [33,2]', 'Electric Potential.Y'),
                'a1z': ('Generic Analog::Electric Potential [33,3]', 'Electric Potential.Z'),
                'a2x': ('Generic Analog::Electric Potential [34,1]', 'Electric Potential.X'),
                'a2y': ('Generic Analog::Electric Potential [34,2]', 'Electric Potential.Y'),
                'a2z': ('Generic Analog::Electric Potential [34,3]', 'Electric Potential.Z'),
                'a3x': ('Generic Analog::Electric Potential [35,1]', 'Electric Potential.X'),
                'a3y': ('Generic Analog::Electric Potential [35,2]', 'Electric Potential.Y'),
                'a3z': ('Generic Analog::Electric Potential [35,3]', 'Electric Potential.Z'),
                'a4x': ('Generic Analog::Electric Potential [36,1]', 'Electric Potential.X'),
                'a4y': ('Generic Analog::Electric Potential [36,2]', 'Electric Potential.Y'),
                'a4z': ('Generic Analog::Electric Potential [36,3]', 'Electric Potential.Z')
            }
    ) -> None:

        self.rawaccel_df = self.extract_analogs(name_mapping)
    
    def extract_rawforce(
            self,
            name_mapping: dict[str,tuple[str,str]] = {
                'fx12': ('Kistler Force Plate 2.0.0.0::Raw [52]', 'Raw.FX12'),
                'fx34': ('Kistler Force Plate 2.0.0.0::Raw [52]', 'Raw.FX34'),
                'fy14': ('Kistler Force Plate 2.0.0.0::Raw [52]', 'Raw.FY14'),
                'fy23': ('Kistler Force Plate 2.0.0.0::Raw [52]', 'Raw.FY23'),
                'fz1': ('Kistler Force Plate 2.0.0.0::Raw [52]', 'Raw.FZ1'),
                'fz2': ('Kistler Force Plate 2.0.0.0::Raw [52]', 'Raw.FZ2'),
                'fz3': ('Kistler Force Plate 2.0.0.0::Raw [52]', 'Raw.FZ3'),
                'fz4': ('Kistler Force Plate 2.0.0.0::Raw [52]', 'Raw.FZ4')
            }
    ) -> None:

        self.rawforce_df = self.extract_analogs(name_mapping)

    def extract_hextrigger(
            self,
            name_mapping: dict[str,tuple[str,str]] = {
                'trig': ('Generic Analog::Electric Potential [47,1]', 'Electric Potential.1')
            }
    ) -> None:
        
        self.hextrigger_df = self.extract_analogs(name_mapping)

    def extract_analogs(
            self,
            name_mapping: dict[str,tuple[str,str]]
    ) -> pd.DataFrame:
        
        analogs_idx = [self.analog_idx[q] for q in name_mapping.values()]
        analogs_data = self.c3d_data["data"]["analogs"][0][analogs_idx]

        analogs_df = pd.DataFrame()
        analogs_df[list(name_mapping.keys())] = analogs_data.T

        return analogs_df

    def segment_trainperts_accelthresh(
            self,
            t_segment: float,
            threshold: float,
            t_base: float = 0.1,
            t_buffer: float = 0.025
    ) -> Tuple[List[npt.NDArray], List[npt.NDArray]]:
        
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

class WindowedSequences(Dataset):
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
        else:
            self.target_tensors = None

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

        input_series = self.input_tensors[series_idx] # shape: (series_length, n_input_channels)
        X_window = input_series[start_idx:end_idx,:] # shape: (window_size, n_input_channels)
        if self.target_tensors is not None:
            target_series = self.target_tensors[series_idx] # shape: (series_length, n_output_channels)
            Y_window = target_series[start_idx:end_idx,:] # shape: (window_size, n_input_channels)
            return X_window, Y_window
        else:
            return X_window

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
        else:
            self.target_tensors = None

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

        input_series = self.input_tensors[series_idx] # shape: (series_length, n_input_channels)
        X_window = input_series[start_idx:end_idx,:].contiguous().flatten() # shape: (n_input_channels*window_size,)
        if self.target_tensors is not None:
            target_series = self.target_tensors[series_idx] # shape: (window_size, n_output_channels)
            Y_window = target_series[start_idx:end_idx,:].contiguous().flatten() # shape: (n_output_channels*window_size,)
            return X_window, Y_window
        else:
            return X_window
        
    def restructure_windowed_output(self, indexed_windows: List[Tuple[int, torch.Tensor]]) -> List[torch.Tensor]:

        num_output_chans = indexed_windows[0][1].numel()//self.window_size
        output_tensors = [torch.zeros(in_tens.shape[0], num_output_chans) for in_tens in self.input_tensors]
        summed_window_segments = [torch.zeros(in_tens.shape[0], num_output_chans) for in_tens in self.input_tensors]
        prediction_counts = [torch.zeros(in_tens.shape[0]) for in_tens in self.input_tensors]

        for index, window in indexed_windows:
            series_idx, start_idx = self.index_map[index]
            end_idx = start_idx + self.window_size
            window_unflattened = window.reshape(-1, num_output_chans)
            summed_window_segments[series_idx][start_idx:end_idx,:] += window_unflattened
            prediction_counts[series_idx][start_idx:end_idx] += 1
        
        for series_idx, series_counts in enumerate(prediction_counts):
            predicted_mask = series_counts != 0
            output_tensors[series_idx][predicted_mask,:] = summed_window_segments[series_idx][predicted_mask,:] / prediction_counts[series_idx][predicted_mask].unsqueeze(1)
            
        return output_tensors
    
class SegmentedSequences(Dataset):
    def __init__(
            self,
            input_arrays: List[npt.ArrayLike],
            target_arrays: Optional[List[npt.ArrayLike]] = None
    ):
        
        self.input_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in input_arrays]
        if target_arrays is not None:
            if len(input_arrays) != len(target_arrays):
                raise ValueError("The number of input series must match the number of target series.")
            self.target_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in target_arrays]
        else:
            self.target_tensors = None

        self.num_segments = len(self.input_tensors)

    def __len__(self):

        return self.num_segments
    
    def __getitem__(self, idx):
        
        X_segment = self.input_tensors[idx]
        sequence_length = X_segment.shape[0]
        if self.target_tensors is not None:
            Y_segment = self.target_tensors[idx]
            return X_segment, Y_segment, sequence_length
        else:
            return X_segment, sequence_length
        
def get_loader(
        dataset_class: Type[Dataset],
        input_arrays: List[npt.ArrayLike],
        batch_size: int,
        shuffle: bool,
        collate_fn: Optional[Any] = None,
        target_arrays: Optional[List[npt.ArrayLike]] = None,
        **dataset_kwargs: Dict[str, Any]
) -> DataLoader:

    dataset = dataset_class(
        input_arrays=input_arrays,
        target_arrays=target_arrays,
        **dataset_kwargs
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return loader
        
def sequence_collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    data_x, data_y, lengths = zip(*batch)

    # pad sequence to the length of the longest sequence in batch
    # batch_first sets output shape to (batch_size, sequence_length, features)
    x_padded = torch.nn.utils.rnn.pad_sequence(data_x, batch_first=True, padding_value=0)
    y_padded = torch.nn.utils.rnn.pad_sequence(data_y, batch_first=True, padding_value=0)

    # sort padded segments according to descending lengths
    lengths = torch.tensor(lengths)
    sorted_lengths, sorted_indices = lengths.sort(descending=True)
    x_padded = x_padded[sorted_indices]
    y_padded = y_padded[sorted_indices]

    return x_padded, y_padded, sorted_lengths

def unpad_unbatch(padded_output: torch.Tensor, lengths: torch.Tensor) -> List[torch.Tensor]:

    unpadded_segments = []
    lengths_list = lengths.cpu().tolist()

    for i, length in enumerate(lengths_list):
        sequence = padded_output[i, :length, :] # padded_output: (B, L, C), using single integer removes that dimension from tensor
        unpadded_segments.append(sequence)

    return unpadded_segments

if __name__ == "__main__":

    TestTrial = C3DMan("data/noLoadPerts_X000_00.c3d")
    TestTrial.print_analog_desclabs()
    TestTrial.extract_rawaccel()
    TestTrial.extract_rawforce()
    TestTrial.extract_hextrigger()

    # TestTrial.accel_df.to_csv("data/accelerations_unloaded_02.csv", index=False)
    # TestTrial.rawforce_df.to_csv("data/forces_unloaded_02.csv", index=False)

    IPython.embed()