
import os
from datetime import datetime
import pandas as pd
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Callable
from format_data import C3DMan, FlattenedWindows, load_perts_h5, calc_standardization_stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import IPython

class BasicMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            input_stats: Tuple[npt.NDArray, npt.NDArray],
            output_stats: Tuple[npt.NDArray, npt.NDArray],
            hidden_dim_1: int = 2048,
            hidden_dim_2: int = 4096
    ):
        super().__init__()

        # register standardization buffers
        input_mean, input_std = input_stats
        self.register_buffer('input_mean', torch.from_numpy(input_mean).float())
        self.register_buffer('input_std', torch.from_numpy(input_std).float())
        output_mean, output_std = output_stats
        self.register_buffer('output_mean', torch.from_numpy(output_mean).float())
        self.register_buffer('output_std', torch.from_numpy(output_std).float())

        self.fc1 = nn.Linear(input_dim, hidden_dim_1) # feature extration/compression
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2) # feature expansion/refinement
        self.fc3 = nn.Linear(hidden_dim_2, output_dim) # final output generation

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # standardize the input
        x_standardized = (x - self.input_mean) / self.input_std

        x_standardized = self.relu(self.fc1(x_standardized)) # through fully connection layer 1, then ReLU activation
        x_standardized = self.relu(self.fc2(x_standardized)) # through fc2, then ReLU activation
        x_standardized = self.fc3(x_standardized) # through final fully connected layer, no activation to predict continuous forces

        # un-standardize the output
        x = (x_standardized * self.output_std) + self.output_mean

        return x # output tensor
    
    def train_val_save(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            val_dataset: FlattenedWindows,
            optimizer,
            num_epochs: int,
            device: torch.device,
            save_path: str
    ) -> float:
        
        best_val_loss = float("inf")
        best_vaf = float("inf")

        print("Training started!")

        for epoch in range(num_epochs):

            # Train Phase
            self.train() # set model to training mode
            running_train_loss = 0.0
            summed_loss = 0.0

            train_loop = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)

            for i, (inputs, targets) in train_loop:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.standardized_mse_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()*inputs.size(0)
                summed_loss += loss.item()
                current_avg_loss = summed_loss/(i+1)

                train_loop.set_postfix(mse=f"{loss.item():.6f}", avg_mse=f"{current_avg_loss:.6f}")

            epoch_train_loss = running_train_loss / len(train_loader.dataset)

            # Validation Phase
            self.eval()
            total_val_loss = 0.0

            indexed_outputs = []
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = self(inputs)
                    loss = self.standardized_mse_loss(outputs, targets)

                    total_val_loss += loss.item()*inputs.size(0)

                    indexed_outputs.append((batch_idx, outputs.squeeze(0)))

            all_outputs = val_dataset.restructure_windowed_output(indexed_outputs)

            epoch_val_loss = total_val_loss / len(val_loader.dataset)
            epoch_val_vaf = calc_avg_vaf(all_outputs, val_dataset.target_tensors)

            # Report and Save Phase
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"    -> Train Loss: {epoch_train_loss:.6f}")
            print(f"    -> Validation Loss: {epoch_val_loss:.6f}")
            print(f"    -> Average Validation VAF: {epoch_val_vaf:.1f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_vaf = epoch_val_vaf

                torch.save(self.state_dict(), save_path)
                print(f"    -> Model saved!")

        return best_vaf
    
    def standardized_mse_loss(self, outputs: torch.Tensor, targets: torch.Tensor, criterion: Optional[Callable] = nn.MSELoss()) -> torch.Tensor:

        # compare standardized prediction to standardized targets
        outputs_standardized = (outputs - self.output_mean) / self.output_std
        targets_standardized = (targets - self.output_mean) / self.output_std

        return criterion(outputs_standardized, targets_standardized)
    
    def load_weights(self, path: str) -> None:

        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval() # set model to evaluation mode

def calc_avg_vaf(outputs: List[torch.Tensor], targets: List[torch.Tensor]) -> float:

    num_channels = targets[0].shape[1]

    segment_avg_vafs = []

    for s in range(len(targets)):

        channel_vafs = []

        for c in range(num_channels):
            y_c = targets[s][:,c]
            y_hat_c = outputs[s][:,c]
            error_c = y_c - y_hat_c
            var_error_c = torch.var(error_c)
            var_targets_c = torch.var(y_c)
            if var_targets_c == 0:
                vaf_c = 100.0 if var_error_c == 0 else -float("inf")
            else:
                vaf_c = (1 - (var_error_c/var_targets_c))*100.0
            channel_vafs.append(vaf_c.item())

        segment_avg_vafs.append(sum(channel_vafs)/len(channel_vafs))

    return sum(segment_avg_vafs)/len(segment_avg_vafs)
    
if __name__ == "__main__":

    window_size = 500
    step_size = 50
    batch_size = 32
    train_ratio = 0.8
    accel_chans = 12
    force_chans = 8
    num_epochs = 20
    learning_rate = 1e-4
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    savefolder = f"models/mlp_models_{timestamp}"
    os.mkdir(savefolder)

    # device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # on Anton, 2 GPUs: cuda:0, cuda:1

    input_dim = accel_chans*window_size
    output_dim = force_chans*window_size

    data_file = "data/noLoadPerts_131125.h5"

    model_info_list = []

    ### TRAIN GENERAL MODEL WITH ALL PERTS

    filename = f"mlp_general.pth"
    savepath = os.path.join(savefolder, filename)

    accel_segs, force_segs, meta_data = load_perts_h5(data_file)

    # split segments for training / validation
    train_pairs, val_pairs = train_test_split(
        list(zip(accel_segs, force_segs)),
        test_size=1.0-train_ratio,
        random_state=42 # seed for reproducibility
    )
    train_accel_segments, train_force_segments = zip(*train_pairs)
    val_accel_segments, val_force_segments = zip(*val_pairs)

    # compute standardization stats on training data
    input_stats, output_stats = calc_standardization_stats(train_accel_segments, train_force_segments)
    flat_input_stats = (np.tile(input_stats[0], window_size), np.tile(input_stats[1], window_size))
    flat_output_stats = (np.tile(output_stats[0], window_size), np.tile(output_stats[1], window_size))

    # create datasets
    train_dataset = FlattenedWindows(
        input_arrays=list(train_accel_segments),
        target_arrays=list(train_force_segments),
        window_size=window_size,
        step_size=step_size
    )

    val_dataset = FlattenedWindows(
        input_arrays=list(val_accel_segments),
        target_arrays=list(val_force_segments),
        window_size=window_size,
        step_size=step_size
    )

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # batch size 1 for sequence reconstruction

    # instantiate model
    Model = BasicMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        input_stats=flat_input_stats,
        output_stats=flat_output_stats
    )
    Model.to(device)

    # train the model
    vaf_saved = Model.train_val_save(
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        optimizer=torch.optim.Adam(Model.parameters(), lr=learning_rate),
        num_epochs=num_epochs,
        device=device,
        save_path=savepath
    )

    model_info_list.append({
        "filename": filename,
        "axis_x": "all",
        "axis_z": "all",
        "dir": "both",
        "vaf": vaf_saved
    })
    
    ### TRAIN MODEL FOR EACH PERTURBATION TYPE
    
    axes_x = [0, 50, 100, 150, 200]
    axes_z = [63, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 96, 100]
    dirs = [-1, 1]

    dir_str = {1: "forward", -1: "reverse"}

    for x in axes_x:

        for z in axes_z:

            for dir in dirs:

                filename = f"mlp_x{x:03d}_z{z:03d}_{dir_str[dir]}.pth"
                savepath = os.path.join(savefolder, filename)

                accel_segs, force_segs, meta_data = load_perts_h5(data_file, [dir], [x], [z])

                # split segments for training / validation
                train_pairs, val_pairs = train_test_split(
                    list(zip(accel_segs, force_segs)),
                    test_size=1.0-train_ratio,
                    random_state=42 # seed for reproducibility
                )
                train_accel_segments, train_force_segments = zip(*train_pairs)
                val_accel_segments, val_force_segments = zip(*val_pairs)

                # compute standardization stats on training data
                input_stats, output_stats = calc_standardization_stats(train_accel_segments, train_force_segments)
                flat_input_stats = (np.tile(input_stats[0], window_size), np.tile(input_stats[1], window_size))
                flat_output_stats = (np.tile(output_stats[0], window_size), np.tile(output_stats[1], window_size))

                # create datasets
                train_dataset = FlattenedWindows(
                    input_arrays=list(train_accel_segments),
                    target_arrays=list(train_force_segments),
                    window_size=window_size,
                    step_size=step_size
                )

                val_dataset = FlattenedWindows(
                    input_arrays=list(val_accel_segments),
                    target_arrays=list(val_force_segments),
                    window_size=window_size,
                    step_size=step_size
                )

                # create dataloaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # batch size 1 for sequence reconstruction

                # instantiate model
                Model = BasicMLP(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    input_stats=flat_input_stats,
                    output_stats=flat_output_stats
                )
                Model.to(device)

                # train the model
                vaf_saved = Model.train_val_save(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    val_dataset=val_dataset,
                    optimizer=torch.optim.Adam(Model.parameters(), lr=learning_rate),
                    num_epochs=num_epochs,
                    device=device,
                    save_path=savepath
                )

                model_info_list.append({
                    "filename": filename,
                    "axis_x": x,
                    "axis_z": z,
                    "dir": dir,
                    "vaf": vaf_saved
                })

    model_info_df = pd.DataFrame(model_info_list)
    model_info_df.to_csv(os.path.join(savefolder, "mlp_model_info.csv"), index=False)