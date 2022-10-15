import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertibleConv1D(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.num_channels = num_channels

        # q : [c, c, 1]  -> kernel size is 1
        q = torch.linalg.qr(torch.randn(num_channels, num_channels))[0].unsqueeze(-1)
        self.W = nn.Parameter(q)

    def forward(self, input, reverse=True):
        # input : [B, C, T]
        _,_, T = input.shape

        if reverse:
            out = F.conv1d(input, self.W)
            # logdet 
            logdet = T*torch.slogdet(self.W.squeeze())[1]
            return out, logdet
        else:
            return F.conv1d(input, self.W.squeeze().inverse().unsqueeze(-1))