"""
DeepONet model definition and masked MSE loss.

Updated version:
    Branch input = SDF field
    Trunk input  = x, y, SDF(x,y)

Important:
    Since SDF(x,y) depends on each geometry, the trunk input is now
    case-dependent and has shape:

        trunk_input: [batch_size, n_points, 3]

    where the three trunk features are:

        [x, y, sdf(x,y)]
"""

import torch
import torch.nn as nn


class SimpleDeepONet(nn.Module):
    def __init__(self, branch_in, trunk_in=3, latent=128):
        super().__init__()

        self.branch = nn.Sequential(
            nn.Linear(branch_in, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, latent),
        )

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent),
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        """
        Parameters
        ----------
        branch_input : tensor
            Shape: [batch_size, branch_in]

        trunk_input : tensor
            Either:
                [n_points, trunk_in]
            or:
                [batch_size, n_points, trunk_in]

        Returns
        -------
        output : tensor
            Shape: [batch_size, n_points]
        """

        b = self.branch(branch_input)

        # --------------------------------------------------------
        # Old style: common trunk coordinates for all cases
        # trunk_input shape = [n_points, trunk_in]
        # --------------------------------------------------------
        if trunk_input.dim() == 2:
            t = self.trunk(trunk_input)
            return torch.einsum("bl,nl->bn", b, t) + self.bias

        # --------------------------------------------------------
        # New style: case-dependent trunk coordinates
        # trunk_input shape = [batch_size, n_points, trunk_in]
        # --------------------------------------------------------
        elif trunk_input.dim() == 3:
            t = self.trunk(trunk_input)
            return torch.einsum("bl,bnl->bn", b, t) + self.bias

        else:
            raise ValueError("trunk_input must have shape [N, d] or [B, N, d].")


def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum() + 1e-12
    return diff2.sum() / denom