
from typing import Tuple, List
from datetime import datetime
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from format_data import WindowedSequences, SegmentedSequences, C3DMan, sequence_collate_fn, get_loader
from cnnlstm_model import calc_standardization_stats
from mlp_model import calc_avg_vaf
import IPython

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

    def loocv(
            self,
            all_trajectories: List[Tuple[npt.ArrayLike, npt.ArrayLike]],
            window_size: int,
            step_size: int,
            batch_size: int,
            save_path: str
    ):
        
        num_folds = len(all_trajectories)
        all_vafs = []
        all_h_firs = []

        for fold in range(num_folds):
            print(f"Running Fold {fold+1}/{num_folds} (Validation on Trajectory {fold+1})")

            val_data = [all_trajectories[fold]]
            train_data = [
                traj for i, traj in enumerate(all_trajectories)
                if i != fold
            ]

            train_inputs, train_targets = zip(*train_data)
            train_window_loader = get_loader(
                input_arrays=train_inputs,
                target_arrays=train_targets,
                dataset_class=WindowedSequences,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=None,
                window_size=window_size, # window size for windowed sequences dataset
                step_size=step_size, # step size for windowed sequences dataset
            )
            train_sequence_loader = get_loader(
                input_arrays=train_inputs,
                target_arrays=train_targets,
                dataset_class=SegmentedSequences,
                batch_size=1,
                shuffle=True,
                collate_fn=sequence_collate_fn
            )

            val_inputs, val_targets = zip(*val_data)
            val_sequence_loader = get_loader(
                input_arrays=val_inputs,
                target_arrays=val_targets,
                dataset_class=SegmentedSequences,
                batch_size=1,
                shuffle=False,
                collate_fn=sequence_collate_fn
            )

            train_loop = tqdm(train_window_loader, desc=f"Running Fold {fold+1}/{num_folds} (Validation on Trajectory {fold+1})", leave=False)

            H_fir = self.fit_mimo_fir(train_window_loop=train_loop)
            self.apply_fir_weights(H_fir)
            all_h_firs.append(H_fir)

            train_vaf = self.eval_vaf(train_sequence_loader)
            val_vaf = self.eval_vaf(val_sequence_loader)
            all_vafs.append(val_vaf)

            print(f"    -> Train VAF: {train_vaf:.1f}%")
            print(f"    -> Validation VAF: {val_vaf:.1f}%")

        avg_val_vaf = sum(all_vafs) / len(all_vafs)
        print(f"LOOCV Complete. Average Validation VAF: {avg_val_vaf:.1f}%")

        stacked_h_firs = torch.stack(all_h_firs, dim=0)
        H_fir_final = torch.mean(stacked_h_firs, dim=0) # average across stacked FIRs
        self.apply_fir_weights(H_fir_final)

        torch.save(self.state_dict(), save_path)
        print("Model saved!")
    
    def train_val_save(
            self,
            train_trajectories: List[Tuple[npt.ArrayLike, npt.ArrayLike]],
            val_trajectories: List[Tuple[npt.ArrayLike, npt.ArrayLike]],
            window_size: int,
            step_size: int,
            save_path: str
    ):
        
        train_inputs, train_targets = zip(*train_trajectories)
        train_window_loader = get_loader(
            input_arrays=train_inputs,
            target_arrays=train_targets,
            dataset_class=WindowedSequences,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=None,
            window_size=window_size, # window size for windowed sequences dataset
            step_size=step_size, # step size for windowed sequences dataset
        )
        train_sequence_loader = get_loader(
            input_arrays=train_inputs,
            target_arrays=train_targets,
            dataset_class=SegmentedSequences,
            batch_size=1,
            shuffle=True,
            collate_fn=sequence_collate_fn
        )

        val_inputs, val_targets = zip(*val_trajectories)
        val_sequence_loader = get_loader(
            input_arrays=val_inputs,
            target_arrays=val_targets,
            dataset_class=SegmentedSequences,
            batch_size=1,
            shuffle=False,
            collate_fn=sequence_collate_fn
        )

        train_loop = tqdm(train_window_loader, desc=f"Training started.", leave=False)

        H_fir = self.fit_mimo_fir(train_window_loop=train_loop)
        self.apply_fir_weights(H_fir)

        self.eval()

        # Test Filter on Training and Validation Data
        train_vaf = self.eval_vaf(train_sequence_loader)
        val_vaf = self.eval_vaf(val_sequence_loader)

        # Report and Save Results
        print(f"Train VAF: {train_vaf:.1f}")
        print(f"Validation VAF: {val_vaf:.1f}")

        torch.save(self.state_dict(), save_path)
        print("Model saved!")

    def eval_vaf(
            self,
            segment_loader: DataLoader
    ):
        
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets, _ in segment_loader:
                outputs = self(inputs)
                all_targets.extend(targets.unbind(0)) # flatten batches to shape (L, C)
                all_outputs.extend(outputs.unbind(0))

        return calc_avg_vaf(all_outputs, all_targets)

    def fit_mimo_fir(
            self,
            train_window_loop: tqdm
    ):
        
        n_coefs = self.input_dim*self.irf_length
        H_vectorized_accum = np.zeros((n_coefs, self.output_dim), dtype=np.float64)
        total_segments_accum = 0

        input_mean_np = self.input_mean.cpu().numpy()
        input_std_np = self.input_std.cpu().numpy()
        output_mean_np = self.output_mean.cpu().numpy()
        output_std_np = self.output_std.cpu().numpy()

        for input_tensors, target_tensors in train_window_loop:
            inputs = input_tensors.numpy() # (batch_size, window_size, input_dim)
            targets = target_tensors.numpy() # (batch_size, window_size, output_dim)

            batch_size, window_size, _ = inputs.shape

            # standardize inputs/outputs
            inputs_standardized = (inputs - input_mean_np) / input_std_np
            targets_standardized = (targets - output_mean_np) / output_std_np

            rows_per_window = window_size - self.irf_length + 1
            total_rows = batch_size * rows_per_window

            Phi_list = []

            for m in range(self.irf_length):
                start_idx = m
                end_idx = window_size - (self.irf_length - 1 - m)
                input_window_shifted = inputs_standardized[:,start_idx:end_idx,:]
                Phi_list.append(input_window_shifted)

            Phi_3D = np.concatenate(Phi_list, axis=2) # (batch_size, rows_per_window, n_coefs=irf_length*input_dim)
            Phi_2D = Phi_3D.reshape(total_rows, n_coefs) # (total_rows, n_coefs)

            target_window_aligned_3D = targets_standardized[:,(self.irf_length-1):,:] # (batch_size, window_size-irf_length, input_dim)
            Y_2D = target_window_aligned_3D.reshape(total_rows, self.output_dim)

            H_vectorized, _, _, _ = np.linalg.lstsq(Phi_2D, Y_2D, rcond=None)

            H_vectorized_accum += H_vectorized
            total_segments_accum += 1

        H_vectorized_avg = H_vectorized_accum/total_segments_accum

        H_reshaped = H_vectorized_avg.reshape(self.irf_length, self.input_dim, self.output_dim)
        H_fir = np.transpose(H_reshaped, (0,2,1)) # (irf_length, output_dim, input_dim)

        return torch.from_numpy(H_fir).float()

if __name__ == "__main__":

    window_size = 50000
    step_size = 10000
    batch_size = 1
    accel_chans = 12
    force_chans = 8
    irf_length = 500
    save_filename_fir = "fir"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    savepath_fir = f"models/{save_filename_fir}_{timestamp}.pth"

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
        accel_segments.append(Set.accel_df.values)
        force_segments.append(Set.rawforce_df.values)

    print("Segments compiled!")

    # compute training data stats for standardization
    accel_stats, force_stats = calc_standardization_stats(
        input_segments=accel_segments,
        output_segments=force_segments
    )

    # instantiate model
    FIRModel = MIMOFIR(
        input_dim=accel_chans,
        output_dim=force_chans,
        input_stats=accel_stats,
        output_stats=force_stats,
        irf_length=irf_length
    )

    FIRModel.loocv(
        all_trajectories=list(zip(accel_segments, force_segments)),
        window_size=window_size,
        step_size=step_size,
        batch_size=batch_size,
        save_path=savepath_fir
    )