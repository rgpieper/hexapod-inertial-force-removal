#!/usr/bin/python

import ezc3d
import c3d
import h5py
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

        with open(self.c3d_path, 'rb') as f:

            reader = c3d.Reader(f)

            print(f"C3D file opened from path:")
            print(c3d_path)

            desc_analog = reader.get("ANALOG").get("DESCRIPTIONS").string_array
            lab_analog = reader.analog_labels
            self.analog_desclab_df = pd.DataFrame({'Description': desc_analog, 'Label': lab_analog})
            self.fs_analog = reader.analog_rate

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

    def extract_convert_accel(
            self,
            channel_indices: List,
            channel_names: List = ['a1x', 'a1y', 'a1z', 'a2x', 'a2y', 'a2z', 'a3x', 'a3y', 'a3z', 'a4x', 'a4y', 'a4z']
    ) -> None:

        assert len(channel_indices) == len(channel_names), f"Number of channel indices ({len(channel_indices)}) does not match number of names ({len(channel_names)})"
        name_mapping = dict(zip(channel_names, channel_indices))
        self.accel_df = self.extract_analogs(name_mapping)
        if self.accel_df.shape[1] > 0:
            print(f"{self.accel_df.shape[1]} acceleration signals extracted with {self.accel_df.shape[0]} datapoints")
        else:
            print(f"ERROR: No acceleration signals found")

    def extract_rawaccel(
            self,
            channel_indices: List,
            channel_names: List = ['a1x', 'a1y', 'a1z', 'a2x', 'a2y', 'a2z', 'a3x', 'a3y', 'a3z', 'a4x', 'a4y', 'a4z']
    ) -> None:

        assert len(channel_indices) == len(channel_names), f"Number of channel indices ({len(channel_indices)}) does not match number of names ({len(channel_names)})"
        name_mapping = dict(zip(channel_names, channel_indices))
        self.rawaccel_df = self.extract_analogs(name_mapping)

        if self.rawaccel_df.shape[1] > 0:
            print(f"{self.rawaccel_df.shape[1]} raw voltage accelerometer signals extracted with {self.rawaccel_df.shape[0]} datapoints")
        else:
            print(f"ERROR: No raw voltage accelerometer signals found")
    
    def extract_rawforce(
            self,
            channel_indices: List,
            channel_names: List = ['fx12', 'fx34', 'fy14', 'fy23', 'fz1', 'fz2', 'fz3', 'fz4']
    ) -> None:

        assert len(channel_indices) == len(channel_names), f"Number of channel indices ({len(channel_indices)}) does not match number of names ({len(channel_names)})"
        name_mapping = dict(zip(channel_names, channel_indices))
        self.rawforce_df = self.extract_analogs(name_mapping)
        if self.rawforce_df.shape[1] > 0:
            print(f"{self.rawforce_df.shape[1]} raw force signals extracted with {self.rawforce_df.shape[0]} datapoints")
        else:
            print(f"ERROR: No raw force signals found")

    def extract_hextrigger(
            self,
            channel_index: int,
            channel_name: str = 'trig'
    ) -> None:
        
        name_mapping = {channel_name: channel_index}
        self.hextrigger_df = self.extract_analogs(name_mapping)
        if self.hextrigger_df.shape[1] == 1:
            print(f"Hexapod trigger signal extracted with {self.rawforce_df.shape[0]} datapoints")
        else:
            print(f"ERROR: Expected 1 hexapod trigger signal, found {self.hextrigger_df.shape[1]}")

    def extract_analogs(
            self,
            name_mapping: dict[str,int]
    ) -> pd.DataFrame:
        
        analogs_idx = [q for q in name_mapping.values()]

        analog_chunks = []
        with open(self.c3d_path, 'rb') as f:
            reader = c3d.Reader(f)
            for frame_idx, points, analogs in reader.read_frames():
                analog_chunks.append(analogs[analogs_idx,:])
        analogs_data = np.hstack(analog_chunks)

        analogs_df = pd.DataFrame()
        analogs_df[list(name_mapping.keys())] = analogs_data.T

        return analogs_df
    
    def segment_perts_trigger(
            self,
            t_segment: float, # sec, segment duration (including buffer)
            t_timeout: float, # sec, time after pulse when new pulses will not be detected
            t_buffer: float = 0.1, # sec, time before trigger to include in segment
            threshold: float = 1.1 # V, trigger high ~1.25V
    ) -> Tuple[List[npt.NDArray], List[npt.NDArray]]:
        
        n_segment = math.ceil(t_segment*self.fs)
        n_timeout = math.ceil(t_timeout*self.fs)
        n_buffer = math.ceil(t_buffer*self.fs)

        accel_segments = []
        force_segments = []
        i = 0
        num_trigs = 0
        while i < len(self.hextrigger_df):
            if self.hextrigger_df.iloc[i].values > threshold:
                num_trigs += 1
                i_start = i - n_buffer if i - n_buffer > 0 else 0
                i_end = i_start + n_segment if i_start + n_segment < len(self.hextrigger_df) else len(self.hextrigger_df)
                accel_seg = self.rawaccel_df.iloc[i_start:i_end].values
                accel_segments.append(accel_seg)
                force_seg = self.rawforce_df.iloc[i_start:i_end].values
                force_segments.append(force_seg)
                i += n_timeout
            else:
                i += 1

        print(f"Found {num_trigs} triggers.")

        return accel_segments, force_segments

    def segment_perts_accelthresh(
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
        
        a_norms["a1n"] = np.linalg.norm(self.rawaccel_df[["a1x", "a1y", "a1z"]].values, axis=1) # accelerometer 1
        a_norms["a2n"] = np.linalg.norm(self.rawaccel_df[["a2x", "a2y", "a2z"]].values, axis=1) # accelerometer 2
        a_norms["a3n"] = np.linalg.norm(self.rawaccel_df[["a3x", "a3y", "a3z"]].values, axis=1) # accelerometer 3
        a_norms["a4n"] = np.linalg.norm(self.rawaccel_df[["a4x", "a4y", "a4z"]].values, axis=1) # accelerometer 4

        a_base = a_norms.iloc[0:n_base].mean()

        accel_segments = []
        force_segments = []
        i = n_base
        while i < len(a_norms):
            if any(abs(a_norms.iloc[i] - a_base) > threshold):
                i_start = i-n_buffer if i-n_buffer > 0 else 0
                i_next = i_start+n_segment if i_start+n_segment < len(a_norms) else len(a_norms)
                accel_seg = self.rawaccel_df.iloc[i_start:i_next].values
                accel_segments.append(accel_seg)
                force_seg = self.rawforce_df.iloc[i_start:i_next].values
                force_segments.append(force_seg)
                i = i_next
            else:
                i += 1

        return accel_segments, force_segments

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
    
class VariableLengthSequences(Dataset):
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

def save_labeled_perts_h5(savepath: str, accel_segs: List[npt.ArrayLike], force_segs: List[npt.ArrayLike], meta_data: List[npt.ArrayLike]) -> None:

    assert len(accel_segs) == len(force_segs) == len(meta_data)

    with h5py.File(savepath, 'w') as f:

        for i, (dir, axis_x, axis_z) in enumerate(meta_data):

            grp = f.create_group(f"pert_{i:05d}")
            grp.create_dataset("accel", data=accel_segs[i])
            grp.create_dataset("force", data=force_segs[i])
            grp.attrs["dir"] = dir
            grp.attrs["axis_x"] = axis_x
            grp.attrs["axis_z"] = axis_z
    
    print(f"Saved {len(meta_data)} perturbations to:")
    print(savepath)

def load_perts_h5(
        filepath: str,
        dirs: Optional[List[int]] = None,
        x_axes: Optional[List[int]] = None,
        z_axes: Optional[List[int]] = None,
) -> Tuple[List[npt.NDArray], List[npt.NDArray], List[Dict]]:
    
    meta_data = []
    accel_segs = []
    force_segs = []

    with h5py.File(filepath, 'r') as f:

        for pert_name in f.keys():

            grp = f[pert_name]

            if (
                (dirs is None or grp.attrs["dir"] in dirs) and
                (x_axes is None or grp.attrs["x_axis"] in x_axes) and
                (z_axes is None or grp.attrs["z_axis"] in z_axes)
            ):
                
                meta_data.append({
                    "name": pert_name,
                    "dir": grp.attrs["dir"],
                    "x_axis": grp.attrs["x_axis"],
                    "z_axis": grp.attrs["z_axis"]
                })
                accel_segs.append(grp["accel"][()])
                force_segs.append(grp["force_segs"][()])
    
    print(f"Loaded {len(meta_data)} perturbations from:")
    print(filepath)

    return accel_segs, force_segs, meta_data

if __name__ == "__main__":

    pertinfo_path = "data/axis_queue_X000.csv"
    pertinfo_df = pd.read_csv(pertinfo_path)
    print(pertinfo_df.head)

    IPython.embed()

    c3d_path = "data/noLoadPerts_X000_01.c3d"
    Trial = C3DMan(c3d_path)
    # Trial.print_analog_desclabs()
    Trial.extract_rawaccel(channel_indices=list(range(62,74)))
    Trial.extract_rawforce(channel_indices=list(range(54,62)))
    Trial.extract_hextrigger(channel_index=74)
    accel_segs, force_segs = Trial.segment_perts_trigger(t_segment=1.3, t_timeout=1.0)

    # IPython.embed()