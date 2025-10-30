from typing import Tuple, List
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from format_data import C3DMan, SegmentedSequences, sequence_collate_fn, unpad_unbatch
from mlp_model import calc_avg_vaf

class MIMOCNNLSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            input_stats: Tuple[npt.NDArray, npt.NDArray],
            output_stats: Tuple[npt.NDArray, npt.NDArray],
            hidden_cnn: int = 128,
            hidden_rnn: int = 156,
            num_rnn_layers: int = 2,
            hidden_lin: int = 64
    ):
        super().__init__()

        self.hidden_rnn = hidden_rnn
        self.num_rnn_layers = num_rnn_layers
        self.output_size = output_size

        # register standardization buffers
        input_mean, input_std = input_stats
        self.register_buffer('input_mean', torch.from_numpy(input_mean).float())
        self.register_buffer('input_std', torch.from_numpy(input_std).float())
        output_mean, output_std = output_stats
        self.register_buffer('output_mean', torch.from_numpy(output_mean).float())
        self.register_buffer('output_std', torch.from_numpy(output_std).float())

        # encoder stage 1: 1D CNN (feature extraction)
        self.conv_encoder = nn.Sequential(
            # input features (e.g. accelerations)
            nn.Conv1d(input_size, hidden_cnn, kernel_size=3, padding=1),
            nn.ReLU(),
            # dilated convolution to increase receptive field without losing length
            nn.Conv1d(hidden_cnn, hidden_cnn, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            # final 1D convolution layer
            nn.Conv1d(hidden_cnn, hidden_cnn, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # encoder stage 2: LSTM (temporal context/memory)
        self.rnn = nn.LSTM(
            input_size=hidden_cnn, # feature size from conv_encoder output
            hidden_size=hidden_rnn,
            num_layers=num_rnn_layers,
            batch_first=True, # input shape: (batch, length, features)
            bidirectional=False # causal
        )

        # dropout stage 3
        self.dropout = nn.Dropout(p=0.2)

        # decoder stage 4: final dense layers (prediction)
        # predict output values (e.g. forces) at each timestep (LSTM outputs for each timstep)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_rnn, hidden_lin),
            nn.ReLU(),
            nn.Linear(hidden_lin, hidden_lin),
            nn.ReLU(),
            nn.Linear(hidden_lin, output_size)
        )

    def forward(self, x_padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

        # standardize the input
        x_standardized = (x_padded - self.input_mean) / self.input_std
        
        batch_size, max_seq_len, _ = x_standardized.shape

        # CNN encoder
        # permute x_padded for 1D CNN: (B, L, C) -> (B, C, L)
        x_permuted = x_standardized.permute(0, 2, 1)
        cnn_out = self.conv_encoder(x_permuted)
        # permute back: (B, C, L) -> (B, L, C)
        cnn_out = cnn_out.permute(0, 2, 1)

        # sequence packing (required for efficient GRU/LSTM on padded data)
        packed_cnn_out = torch.nn.utils.rnn.pack_padded_sequence(
            cnn_out,
            lengths.cpu(),
            batch_first=True
        )

        # initialize hidden state (h0) and cell state (c0) to zeros: (num_rnn_layers*num_directions), batch_size, hidden_rnn)
        h0 = torch.zeros(self.num_rnn_layers*1, batch_size, self.hidden_rnn).to(x_standardized.device)
        c0 = torch.zeros(self.num_rnn_layers*1, batch_size, self.hidden_rnn).to(x_standardized.device)

        # LSTM forward pass modeling memory
        packed_rnn_out, _ = self.rnn(packed_cnn_out, (h0, c0)) # hidden state and cell state start at zero for each segment, thanks to pack_padded_sequence

        # unpack sequence: (batch, L, hidden_rnn)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_rnn_out,
            batch_first=True,
            total_length=max_seq_len
        )

        # apply dropout
        rnn_out = self.dropout(rnn_out)

        # decoder predicting force at each time step
        # reshape to padded sequence shape
        final_output_standardized = self.decoder(rnn_out.reshape(-1, self.hidden_rnn))
        final_output_standardized = final_output_standardized.reshape(batch_size, max_seq_len, self.output_size)

        # un-standardize the output
        final_output = (final_output_standardized * self.output_std) + self.output_mean

        return final_output
    
    def masked_mse_loss(self, outputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

        # compare standardized prediction to standardized targets
        outputs_standardized = (outputs - self.output_mean) / self.output_std
        targets_standardized = (targets - self.output_mean) / self.output_std

        # compute squared error for all elements
        squared_error = (outputs_standardized - targets_standardized)**2

        # create mask
        L_max = outputs.size(1)
        sequence_mask = torch.arange(L_max, device=outputs.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = sequence_mask.unsqueeze(2).expand_as(squared_error).float()
        
        # apply mask
        masked_error = squared_error * mask

        # compute MSE over valid (non-padding) elements
        sum_squared_error = torch.sum(masked_error)
        total_valid_elements = torch.sum(lengths)*outputs.size(2) # lengths * num_channels
        return sum_squared_error/total_valid_elements
    
    def train_val_save(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer,
            num_epochs: int,
            device: torch.device,
            save_path: str
    ) -> None:
        
        best_val_loss = float("inf")

        print("Training started!")

        for epoch in range(num_epochs):

            # Train Phase
            self.train()
            running_train_loss = 0.0
            total_train_elements = 0

            for inputs, targets, lengths in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs, lengths) # pass lengths to model
                loss = self.masked_mse_loss(outputs, targets, lengths) # calculate masked loss
                loss.backward()
                optimizer.step()

                total_valid_elements = torch.sum(lengths)*outputs.size(2)
                total_train_elements += total_valid_elements.item()
                running_train_loss += loss.item() * total_valid_elements.item()

            epoch_train_loss = running_train_loss / total_train_elements

            # Validation Phase
            self.eval()
            total_val_loss = 0.0
            total_val_elements = 0

            val_targets = []
            val_outputs = []

            with torch.no_grad():
                for inputs, targets, lengths in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = self(inputs, lengths)
                    loss = self.masked_mse_loss(outputs, targets, lengths)

                    total_valid_elements = torch.sum(lengths)*outputs.size(2)
                    total_val_elements += total_valid_elements
                    total_val_loss += loss.item()*total_valid_elements.item()

                    unpadded_targets = unpad_unbatch(targets.cpu(), lengths)
                    unpadded_outputs = unpad_unbatch(outputs.cpu(), lengths)
                    val_targets.extend(unpadded_targets)
                    val_outputs.extend(unpadded_outputs)

            epoch_val_loss = total_val_loss / total_val_elements
            epoch_val_vaf = calc_avg_vaf(val_outputs, val_targets)

            # Report and Save Phase
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"    -> Train Loss: {epoch_train_loss:.6f}")
            print(f"    -> Validation Loss: {epoch_val_loss:.6f}")
            print(f"    -> Average Validation VAF: {epoch_val_vaf:.1f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

                torch.save(self.state_dict(), save_path)
                print(f"    -> Model saved!")
    
    def load_weights(self, path: str) -> None:

        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval() # set model to evaluation mode

def calc_standardization_stats(
        input_segments: List[npt.ArrayLike],
        output_segments: List[npt.ArrayLike]
) -> Tuple[Tuple[npt.NDArray, npt.NDArray], Tuple[npt.NDArray, npt.NDArray]]:
    
    all_inputs = np.concatenate(input_segments, axis=0)
    all_outputs = np.concatenate(output_segments, axis=0)

    input_mean = all_inputs.mean(axis=0, dtype=np.float32)
    input_std = all_inputs.std(axis=0, dtype=np.float32)
    input_std[input_std == 0] = 1.0 # handle zero-variance channels

    output_mean = all_outputs.mean(axis=0, dtype=np.float32)
    output_std = all_outputs.std(axis=0, dtype=np.float32)
    output_std[output_std == 0] = 1.0 # handle zero-variance channels

    return (input_mean, input_std), (output_mean, output_std)

if __name__ == "__main__":

    accel_chans = 12
    force_chans = 8
    train_ratio = 0.8
    batch_size = 16
    num_epochs = 20

    C3D_datasets = (
        C3DMan("data/fullgrid_loaded_01.c3d"),
        # C3DMan("data/fullgrid_unloaded_01.c3d"),
        # C3DMan("data/fullgrid_unloaded_02.c3d")
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
    Model = MIMOCNNLSTM(
        input_size=accel_chans,
        output_size=force_chans,
        input_stats=accel_stats,
        output_stats=force_stats
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sequence_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=sequence_collate_fn)

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model.to(device)

    # setup optimizer
    # learning_rate = 1e-4
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate)

    # train the model
    Model.train_val_save(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_path="models/cnnlstm_29102025.pth"
    )