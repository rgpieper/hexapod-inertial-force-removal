
import os
from datetime import datetime
from typing import Tuple, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import numpy.typing as npt
from format_data import WindowedSequences, load_perts_h5, calc_standardization_stats
from mlp_model import calc_avg_vaf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import IPython

class MIMOLSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            input_stats: Tuple[npt.NDArray, npt.NDArray],
            output_stats: Tuple[npt.NDArray, npt.NDArray],
            hidden_rnn: int = 64,
            num_layers_rnn: int = 2,
            hidden_lin: int = 64
    ):
        super().__init__()

        self.hidden_rnn = hidden_rnn
        self.num_layers_rnn = num_layers_rnn
        self.output_size = output_size

        # register standardization buffers
        input_mean, input_std = input_stats
        self.register_buffer('input_mean', torch.from_numpy(input_mean).float())
        self.register_buffer('input_std', torch.from_numpy(input_std).float())
        output_mean, output_std = output_stats
        self.register_buffer('output_mean', torch.from_numpy(output_mean).float())
        self.register_buffer('output_std', torch.from_numpy(output_std).float())

        self.rnn_directions = 2 # 1: unidirectional (causal), 2: bidirectional (non-causal)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_rnn,
            num_layers=num_layers_rnn,
            batch_first=True, # input shape: (batch, length, features)
            bidirectional=True if self.rnn_directions == 2 else False
        )

        self.dropout = nn.Dropout(p=0.2) # dropout for robust training, preventing overfitting (auto-disabled for model.eval())

        self.decoder = nn.Sequential(
            nn.Linear(hidden_rnn*self.rnn_directions, hidden_lin), # rnn output size is double if bidirectional
            nn.ReLU(),
            nn.Linear(hidden_lin, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_standardized = (x - self.input_mean) / self.input_std

        batch_size, window_length, _ = x_standardized.shape

        # initialize hidden state (h0) and cell state (c0) to zeros: (num_rnn_layers*num_directions), batch_size, hidden_rnn)
        h0 = torch.zeros(self.num_rnn_layers*self.rnn_directions, batch_size, self.hidden_rnn).to(x_standardized.device)
        c0 = torch.zeros(self.num_rnn_layers*self.rnn_directions, batch_size, self.hidden_rnn).to(x_standardized.device)

        # LSTM layers
        rnn_out, _ = self.rnn(x_standardized, (h0, c0)) # hidden state and cell state start at zero for each segment, thanks to pack_padded_sequence

        # dropout layer
        rnn_out = self.dropout(rnn_out)

        # linear decoder layers
        rnn_out_flat = rnn_out.reshape(-1, self.hidden_rnn*self.rnn_directions) # flatten for linear layers
        final_out_flat_standardized = self.decoder(rnn_out_flat)
        final_out_standardized = final_out_flat_standardized.reshape(batch_size, window_length, self.output_size) # unflatten -> (B, L, C)

        # un-standardize output
        final_out = (final_out_standardized * self.output_std) + self.output_mean

        return final_out
    
    def train_val_save(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            val_dataset: WindowedSequences,
            optimizer,
            num_epochs: int,
            device: torch.device,
            save_path:str
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
    savefolder = f"models/lstm_models_{timestamp}"
    os.mkdir(savefolder)

    # device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # on Anton, 2 GPUs: cuda:0, cuda:1

    data_file = "data/noLoadPerts_131125.h5"

    model_info_list = []

    ### TRAIN GENERAL MODEL WITH ALL PERTS

    filename = f"lstm_general.pth"
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

    # create datasets
    train_dataset = WindowedSequences(
        input_arrays=list(train_accel_segments),
        target_arrays=list(train_force_segments),
        window_size=window_size,
        step_size=step_size
    )

    val_dataset = WindowedSequences(
        input_arrays=list(val_accel_segments),
        target_arrays=list(val_force_segments),
        window_size=window_size,
        step_size=step_size
    )

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # batch size 1 for sequence reconstruction

    Model = MIMOLSTM(
        input_size=accel_chans,
        output_size=force_chans,
        input_stats=input_stats,
        output_stats=output_stats,
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

                filename = f"lstm_x{x:03d}_z{z:03d}_{dir_str[dir]}.pth"
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

                # create datasets
                train_dataset = WindowedSequences(
                    input_arrays=list(train_accel_segments),
                    target_arrays=list(train_force_segments),
                    window_size=window_size,
                    step_size=step_size
                )

                val_dataset = WindowedSequences(
                    input_arrays=list(val_accel_segments),
                    target_arrays=list(val_force_segments),
                    window_size=window_size,
                    step_size=step_size
                )

                # create dataloaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # batch size 1 for sequence reconstruction

                # instantiate model
                Model = MIMOLSTM(
                    input_size=accel_chans,
                    output_size=force_chans,
                    input_stats=input_stats,
                    output_stats=output_stats
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
    model_info_df.to_csv(os.path.join(savefolder, "lstm_model_info.csv"), index=False)