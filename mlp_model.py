import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from format_data import C3DMan, FlatAccForce
import IPython

class AccForceMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim_1: int = 2048,
            hidden_dim_2: int = 4096
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim_1) # feature extration/compression
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2) # feature expansion/refinement
        self.fc3 = nn.Linear(hidden_dim_2, output_dim) # final output generation

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):

        x = self.relu(self.fc1(x)) # through fully connection layer 1, then ReLU activation
        x = self.relu(self.fc2(x)) # through fc2, then ReLU activation
        x = self.fc3(x) # through final fully connected layer, no activation to predict continuous forces

        return x # output tensor
    
    def train_val_save(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion,
            optimizer,
            num_epochs: int,
            device: torch.device,
            save_path: str
    ):
        
        best_val_loss = float("inf")

        print("Training started!")

        for epoch in range(num_epochs):

            # Train Phase
            self.train() # set model to training mode
            running_train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()*inputs.size(0)

            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            epoch_train_vaf = calc_avg_vaf(outputs, targets, num_channels=8)

            # Validation Phase
            self.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = self(inputs)
                    loss = criterion(outputs, targets)

                    total_val_loss += loss.item()*inputs.size(0)

            epoch_val_loss = total_val_loss / len(val_loader.dataset)
            epoch_val_vaf = calc_avg_vaf(outputs, targets, num_channels=8)

            # Report and Save Phase
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"    -> Train Loss: {epoch_train_loss:.6f}, Avg VAF: {epoch_train_vaf:.1f}%")
            print(f"    -> Validation Loss: {epoch_val_loss:.6f}, Avg VAF: {epoch_val_vaf:.1f}%")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

                torch.save(self.state_dict(), save_path)
                print(f"    -> Model saved!")
    
    def load_weights(self, path: str):

        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval() # set model to evaluation mode

def calc_avg_vaf(outputs: torch.Tensor, targets: torch.Tensor, num_channels: int):

    channel_outputs = outputs.reshape(-1, num_channels)
    channel_targets = targets.reshape(-1, num_channels)

    channel_vafs = []

    for c in range(num_channels):
        y_c = channel_targets[:,c]
        y_hat_c = channel_outputs[:,c]
        error_c = y_c - y_hat_c
        var_error_c = torch.var(error_c)
        var_targets_c = torch.var(y_c)
        if var_targets_c == 0:
            vaf_c = 100.0 if var_error_c == 0 else -float("inf")
        else:
            vaf_c = (1 - (var_error_c/var_targets_c))*100.0
        channel_vafs.append(vaf_c.item())
    
    return sum(channel_vafs)/len(channel_vafs)
    
if __name__ == "__main__":

    window_size = 500
    step_size = 50
    batch_size = 32
    train_split = 0.8
    accel_chans = 12
    force_chans = 8

    input_dim = accel_chans*window_size
    output_dim = force_chans*window_size

    Model = AccForceMLP(input_dim=input_dim, output_dim=output_dim)

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

    # create dataset
    full_dataset = FlatAccForce(
        accel_arrays=accel_segments,
        force_arrays=force_segments,
        window_size=window_size,
        step_size=step_size
    )

    del C3D_datasets, accel_segments, force_segments
    
    # split training and validation datasets
    train_size = int(train_split*len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model.to(device)

    # setup loss function
    criterion = nn.MSELoss() # mean squared error

    # setup optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate)
    num_epochs = 20

    # train the model
    Model.train_val_save(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_path="models/mlp_28102025.pth"
    )