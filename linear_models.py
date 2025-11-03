
from typing import Tuple, List
from datetime import datetime
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from format_data import SegmentedSequences, C3DMan
from cnnlstm_model import calc_standardization_stats
from mlp_model import calc_avg_vaf

class MIMOFIR(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            input_stats: Tuple[npt.NDArray, npt.NDArray],
            output_stats: Tuple[npt.NDArray, npt.NDArray],
            irf_length: int
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.irf_length = irf_length

        # register standardization buffers
        input_mean, input_std = input_stats
        self.register_buffer('input_mean', torch.from_numpy(input_mean).float())
        self.register_buffer('input_std', torch.from_numpy(input_std).float())
        output_mean, output_std = output_stats
        self.register_buffer('output_mean', torch.from_numpy(output_mean).float())
        self.register_buffer('output_std', torch.from_numpy(output_std).float())

        self.fir_filter = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            kernel_size=self.irf_length,
            padding=0, # no internal padding, input padded manually for causal prediction
            bias=False # standard FIR model has no bias term
        )
        self.fir_filter.weight.requires_grad_(False) # layer is a fixed linear model by default

    def forward(
            self,
            input: torch.Tensor # shape: (batch, channels, length)
    ):
        
        input_standardized = (input - self.input_mean) / self.input_std
        input_permuted = input_standardized.transpose(1, 2) # (B, L, C) -> (B, C, L) for Conv1d
        
        # apply causal padding
        input_padded = nn.functional.pad(
            input_permuted,
            pad=(self.irf_length-1, 0), # pad only the left side
            mode="constant",
            value=0.0
        )
        
        # apply fixed FIR filter (convolution)
        # reshape to original
        output_standardized = self.fir_filter(input_padded)
        output_standardized = output_standardized.transpose(1, 2) # (B, C, L) -> (B, L, C)

        # un-standardize the output
        return (output_standardized * self.output_std) + self.output_mean

    def apply_fir_weights(
            self,
            H_fir: torch.Tensor # (irf_length, output_dim, input_dim)
    ):
        
        H_weights = H_fir.permute(1,2,0).contiguous() # (output_dim, input_dim, irf_length)
        
        with torch.no_grad():
            self.fir_filter.weight.copy_(H_weights)
            self.fir_filter.weight.requires_grad_(False) # freeze weights so layer is a fixed linear model

    def train_val_save(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            window_size: int,
            step_size: int,
            save_path: str
    ):
        
        # Fit and Apply Filter
        H_fir = self.fit_mimo_fir(train_loader, window_size, step_size)
        self.apply_fir_weights(H_fir)

        self.eval()

        # Test Filter on Training and Validation Data
        train_vaf = self.eval_vaf(train_loader)
        val_vaf = self.eval_vaf(val_loader)

        # Report and Save Results
        print(f"Train VAF: {train_vaf:.1f}")
        print(f"Validation VAF: {val_vaf:.1f}")

        torch.save(self.state_dict(), save_path)
        print("Model saved!")

    def eval_vaf(
            self,
            loader: DataLoader
    ):
        
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets, _ in loader:
                outputs = self(inputs)
                all_targets.extend(targets.unbind(0)) # flatten batches to shape (L, C)
                all_outputs.extend(outputs.unbind(0))

        return calc_avg_vaf(all_outputs, all_targets)

    def fit_mimo_fir(
            self,
            train_loader: DataLoader,
            window_size: int,
            step_size: int
    ):
        
        n_coefs = self.input_dim*self.irf_length
        H_vectorized_accum = np.zeros((n_coefs, self.output_dim), dtype=np.float64)
        total_segments_accum = 0

        for input_tensors, target_tensors, lengths in train_loader:
            inputs = input_tensors.numpy() # (batch_size, segment_length, input_dim)
            targets = target_tensors.numpy() # (batch_size, segment_length, output_dim)

            # standardize inputs/outputs
            input_standardized = (inputs - self.input_mean.cpu().numpy()) / self.input_std.cpu().numpy()
            targets_standardized = (targets - self.output_mean.cpu().numpy()) / self.output_std.cpu().numpy()

            batch_size = input_standardized.shape[0]
            
            for i in range(batch_size):
                series_length = lengths[i].item()
                max_start = series_length - window_size
                start_indices = np.arange(0, series_length-window_size+step_size, step_size)
                valid_start_indices = start_indices[start_indices <= max_start]

                for start_idx in valid_start_indices:
                    end_idx = start_idx + window_size

                    input_window = input_standardized[i,start_idx:end_idx,:] # (window_size, input_dim)
                    target_window = targets_standardized[i,start_idx:end_idx,:] # (window_size, output_dim)

                    Phi_list = []

                    for m in range(self.irf_length):
                        input_window_shifted = input_window[(self.irf_length-1-m):(window_size-m),:]
                        Phi_list.append(input_window_shifted)

                    Phi = np.concatenate(Phi_list, axis=1)

                    target_window_aligned = target_window[(self.irf_length-1):,:]

                    H_vectorized, _, _, _ = np.linalg.lstsq(Phi, target_window_aligned, rcond=None)

                    H_vectorized_accum += H_vectorized
                    total_segments_accum += 1

        H_vectorized_avg = H_vectorized_accum/total_segments_accum

        H_reshaped = H_vectorized_avg.reshape(self.irf_length, self.input_dim, self.output_dim)
        H_fir = np.transpose(H_reshaped, (0,2,1)) # (irf_length, output_dim, input_dim)

        return torch.from_numpy(H_fir).float()

if __name__ == "__main__":

    window_size = 3000
    step_size = 300
    batch_size = 32
    train_ratio = 0.8
    accel_chans = 12
    force_chans = 8
    irf_length = 500
    save_filename_fir = "fir"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    savepath_fir = f"models/{save_filename_fir}_{timestamp}.pth"

    input_dim = accel_chans*window_size
    output_dim = force_chans*window_size

    C3D_datasets = (
        C3DMan("data/fullgrid_loaded_01.c3d"),
        C3DMan("data/fullgrid_unloaded_01.c3d"),
        C3DMan("data/fullgrid_unloaded_02.c3d")
    )

    accel_segments = []
    force_segments = []
    for Set in C3D_datasets:
        Set.extract_accel()
        Set.extract_rawforce()
        accel_segs, rawforce_segs = Set.segment_trainperts(t_segment = 1.15, threshold=1.0)
        accel_segments.extend(accel_segs)
        force_segments.extend(rawforce_segs)

    print("Segments compiled!")

    # split pert segments for training/validation
    segment_pairs = list(zip(accel_segments, force_segments))
    train_pairs, val_pairs = train_test_split(
        segment_pairs,
        test_size=1.0-train_ratio,
        random_state=42 # seed for reproducibility
    )
    train_accel_segments, train_force_segments = zip(*train_pairs)
    val_accel_segments, val_force_segments = zip(*val_pairs)
    train_accel_segments = list(train_accel_segments)
    train_force_segments = list(train_force_segments)
    val_accel_segments = list(val_accel_segments)
    val_force_segments = list(val_force_segments)

    # compute training data stats for standardization
    accel_stats, force_stats = calc_standardization_stats(
        input_segments=train_accel_segments,
        output_segments=train_force_segments
    )

    # instantiate model
    FIRModel = MIMOFIR(
        input_dim=accel_chans,
        output_dim=force_chans,
        input_stats=accel_stats,
        output_stats=force_stats,
        irf_length=irf_length
    )

    # create datasets
    train_dataset = SegmentedSequences(
        input_arrays=train_accel_segments,
        target_arrays=train_force_segments
    )

    val_dataset = SegmentedSequences(
        input_arrays=val_accel_segments,
        target_arrays=val_force_segments
    )

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    FIRModel.train_val_save(
        train_loader=train_loader,
        val_loader=val_loader,
        window_size=window_size,
        step_size=step_size,
        save_path=savepath_fir
    )