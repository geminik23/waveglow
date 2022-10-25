import torch
import torch.nn as nn
import torch.nn.functional as F


# WN() uses layers of dilated convolutions with gatedtanh nonlinearities
# our convolutions have 3 taps and are not causal.
# gated-tanh nonlinearites of each layer as in WaveNet

class GatedNonCausualLayer(nn.Module):
    """
    modification from my wavenet implementation
    for residuals and skips, use one conv
    """
    def __init__(self, residual_channels, skip_channels, condition_channels, kernel_size=3, dilation=1):
        super().__init__()

        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.condition_channels = condition_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.hidden_channels = residual_channels 

        self.padding = ((kernel_size-1)*dilation)//2
        self.input_conv = nn.Conv1d(residual_channels, condition_channels, kernel_size=kernel_size, dilation=dilation, padding=self.padding)
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.out_conv = nn.Conv1d(condition_channels, residual_channels + skip_channels, kernel_size=1)

    def forward(self, input, cond):
        """
        B: batch_size, R: residual_channel, T: time series
        C: condition_channel, S: skip_channel
        input : [B, R, T]
        condition : [B, C, T] 
        """

        # [B, R, T] -> [B, C, T]
        x = self.input_conv(input) + cond
        gates = torch.mul( self.tanh(x), self.sigm(x))
        # [B, C, T] -> [B, R+S, T] -> [B,R,T], [B,S,T]
        return self.out_conv(gates).split([self.residual_channels, self.skip_channels], dim=1)
        


#
class WaveNetLike(nn.Module):
    def __init__(self, in_channels, mel_channels, residual_channels, skip_channels, hidden_channels, kernel_size=3, max_dilation=8):
        super().__init__()
        # no details of WN layer structure from paper. 
        # refered from official implemenetation.(https://github.com/NVIDIA/waveglow)

        self.residual_channels = residual_channels
        self.skip_channels = skip_channels

        self.hidden_channels = hidden_channels # condition_channel in GatedNonCausalLayer
        self.kernel_size = kernel_size
        self.max_dilation = max_dilation

        dilations = [2**d for d in range(0, max_dilation+1)]
        self.n_layer = len(dilations)

        self.pre_conv = nn.Conv1d(in_channels, residual_channels, 1)
        self.cond_conv = nn.Conv1d(mel_channels, hidden_channels*self.n_layer, 1)

        layers = []
        for d in dilations:
            layers.append(GatedNonCausualLayer(residual_channels, skip_channels, hidden_channels, kernel_size, dilation=d))
        self.layers = nn.ModuleList(layers)

        self.end = torch.nn.Conv1d(skip_channels, 2*in_channels, 1)
        self.end.weight.data.zero_()
        self.end.bias.data.zero_()

    def forward(self, x, cond):
        # x : [B, in_channels, T]
        # cond : [B, mel_channels, T]

        # [B, M, T] -> n_layer * [B, hidden_channels, T]
        conds = self.cond_conv(cond).split(self.hidden_channels, 1)

        residual, skips = self.pre_conv(x), []
        for n, layer in enumerate(self.layers):
            residual, skip = layer(residual, conds[n])
            skips.append(skip)
        # [B, skip_channels, T]
        skip = torch.stack(skips).sum(0)
        # [B, in_channels*2, T] 
        return self.end(skip)


class AffineCoupling(nn.Module):
    def __init__(self, WNObject, *args, **kargs):
        super().__init__()
        self.wn = WNObject(*args, **kargs)

    def forward(self, input, cond, reverse=False):
        if reverse: 
            ## reverse
            # y_a, y_b = split(y)
            # los_s, t = wn(y_a, cond)
            # x_b =(y_b - t) / s
            # x_a = y_a
            # concat(x_a, x_b)
            y = input
            ya, yb = y.chunk(2, 1)
            out = self.wn(ya, cond)
            log_s, t = out.chunk(2, 1)
            xb = (yb - t) / torch.exp(log_s)
            xa = ya
            x = torch.cat((xa, xb), 1)
            return x, -log_s
        else: 
            ## forward
            # x_a, x_b = split(x)
            # log_s, t = wn(x_a, cond)
            # x_b' = s*x_b + t
            # concat(x_a, x_b')
            x = input
            xa, xb = x.chunk(2, 1)
            za = xa
            out = self.wn(xa, cond)
            log_s, t = out.chunk(2, 1)
            zb = torch.exp(log_s)*xb + t
            z = torch.cat((za, zb), 1)
            return z, log_s



class InvertibleConv1D(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.num_channels = num_channels

        # q : [c, c, 1]  -> kernel size is 1
        q = torch.linalg.qr(torch.randn(num_channels, num_channels))[0].unsqueeze(-1)
        self.W = nn.Parameter(q)

    def forward(self, input, reverse=False):
        # input : [B, C, T]
        _,_, T = input.shape

        if reverse:
            return F.conv1d(input, self.W.squeeze().inverse().unsqueeze(-1))
        else:
            out = F.conv1d(input, self.W)
            # logdet 
            logdet = T*torch.slogdet(self.W.squeeze())[1]
            return out, logdet
