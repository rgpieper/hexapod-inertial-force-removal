
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional
import numpy.typing as npt
from tqdm import tqdm
from nfoursid.nfoursid import NFourSID

class MIMOSS(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            input_stats: Tuple[npt.NDArray, npt.NDArray],
            output_stats: Tuple[npt.NDArray, npt.NDArray],
            system_order: int
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.system_order = system_order

        # register standardization buffers
        input_mean, input_std = input_stats
        self.register_buffer('input_mean', torch.from_numpy(input_mean).float())
        self.register_buffer('input_std', torch.from_numpy(input_std).float())
        output_mean, output_std = output_stats
        self.register_buffer('output_mean', torch.from_numpy(output_mean).float())
        self.register_buffer('output_std', torch.from_numpy(output_std).float())

        self.register_buffer('A', torch.zeros(system_order, system_order, dtype=torch.float32))
        self.register_buffer('B', torch.zeros(system_order, input_dim, dtype=torch.float32))
        self.register_buffer('C', torch.zeros(output_dim, system_order, dtype=torch.float32))
        self.register_buffer('D', torch.zeros(output_dim, input_dim, dtype=torch.float32))
        self.register_buffer('initial_state', torch.zeros(system_order, dtype=torch.float32))

    def forward(
            self,
            input: torch.Tensor # shape: (batch, length, channels)
    ) -> torch.Tensor:
        
        input_standardized = (input - self.input_mean) / self.input_std

        batch_size, seq_len, _ = input_standardized.shape

        # initialize state
        X = self.initial_state.unsqueeze(0).repeat(batch_size, 1)

        # output storage: (batch, length, output_dim)
        Y_standardized = torch.zeros(batch_size, seq_len, self.output_dim, device=input.device)

        for k in range(seq_len):

            u_k = input_standardized[:,k,:] # (batch, output_dim)

            # state update
            # X_k+1 = A * X_k + B * u_k
            X_new = torch.matmul(X, self.A.T) + torch.matmul(u_k, self.B.T)

            # output computation
            # y_k = C * X_k + D * u_k
            y_k = torch.matmul(X, self.C.T) + torch.matmul(u_k, self.D.T)

            Y_standardized[:,k,:] = y_k

            X = X_new

        return (Y_standardized * self.output_std) + self.output_mean
    
    def apply_ss_matrices(
            self,
            A: torch.Tensor,
            B: torch.Tensor,
            C: torch.Tensor,
            D: torch.Tensor,
            initial_state: Optional[torch.Tensor] = None
    ):
        
        with torch.no_grad():
            self.A.copy_(A.float())
            self.B.copy_(B.float())
            self.C.copy_(C.float())
            self.D.copy_(D.float())

            if initial_state is not None:
                self.initial_state.copy_(initial_state.float())

    