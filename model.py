from cgi import print_arguments
import torch
import torch.nn as nn
from modules import InvertibleConv1D, AffineCoupling, WaveNetLike

class WaveGlow(nn.Module):
    def __init__(self, n_mels, n_flows, n_groups, n_early_every, n_early_size, win_length=1024, hop_length=256, sigma=1.0, wn_config:dict={}):
        super().__init__()
        self.n_mels = n_mels
        self.n_flows = n_flows
        self.n_groups = n_groups
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.sigma = sigma

        self.win_length = win_length
        self.hop_length = hop_length

        # WN args
        self.wn_config = wn_config


        # upsample for melspec
        self.upsample = nn.utils.weight_norm(nn.ConvTranspose1d(n_mels, n_mels, kernel_size=win_length, stride=hop_length))

        self.inv1x1 = nn.ModuleList()
        self.affines = nn.ModuleList()

        n_channels = n_groups
        for i in range(n_flows):
            if i % self.n_early_every == 0 and i:
                n_channels -= n_early_size
            self.inv1x1.append(InvertibleConv1D(n_channels))
            self.affines.append(AffineCoupling(WaveNetLike, in_channels=n_channels//2, mel_channels=n_mels*n_groups, residual_channels=wn_config['residual_channels'], skip_channels=wn_config['skip_channels'], hidden_channels=wn_config['hidden_channels'], kernel_size=wn_config['kernel_size'], max_dilation=wn_config['max_dilation'] ))

        self.start_n_channels = n_channels
    

    def forward(self, input):
        x, mels = input
        x = x.float()
        # x: [B, T]
        # mels : [B, n_mels, n_frames]

        batch_size = x.size(0)

        ## upsample the mels ready for the condition
        mels = self.upsample(mels) 
        if mels.size(-1) > x.size(-1):
            # set same length of audio
            mels = mels[..., :x.size(-1)]


        ## squeeze to vector
        # unfold the audio and mels
        # [B, T] -> [B, T//G, G] -> [B, G, T//G]
        x = x.unfold(1, self.n_groups, self.n_groups).permute(0, 2, 1) 

        # [B, M, T] -> [B, T//G, M, G] -> [B, G*M, T//G]
        mels = mels.unfold(2, self.n_groups, self.n_groups).permute(0, 2, 1, 3)
        mels = mels.reshape(batch_size, mels.size(1), -1).permute(0, 2, 1)

        output= []
        ldj = torch.tensor([0.0]).to(x.device)

        for i in range(self.n_flows):
            if i % self.n_early_every == 0 and i:
                early_output = (x[:,:self.n_early_size,:])
                x = x[:,self.n_early_size:,:]
                output.append(early_output)
            
            x, logdet_w = self.inv1x1[i](x)
            x, log_s = self.affines[i](x, mels)
            ldj += logdet_w + log_s.sum()
    
        z = torch.cat(output, 1)
        loss = (z*z).sum()/(2*self.sigma*self.sigma) - ldj
        return loss



    def inference(self, mels, sigma=0.6, device='cpu'):
        batch_size = mels.size(0)

        # mels : [B, n_mels, n_frames]
        mels = mels.to(device)
        mels = self.upsample(mels)
        mels = mels[..., :-(self.win_length - self.hop_length)]
    
        # [B, M, T] -> [B, T//G, M, G] -> [B, G*M, T//G]
        mels = mels.unfold(2, self.n_groups, self.n_groups).permute(0, 2, 1, 3)
        mels = mels.reshape(batch_size, mels.size(1), -1).permute(0, 2, 1)

        x = torch.empty((batch_size, self.start_n_channels, mels.size(-1)), device=device).normal_(std=sigma)

        for i in range(self.n_flows-1, -1, -1):
            x, _ = self.affines[i](x, mels, reverse=True)
            x = self.inv1x1[i](x, reverse=True)

            if i % self.n_early_every == 0 and i>0:
                z = torch.empty((batch_size, self.n_early_size, mels.size(-1)), device=device).normal_(std=sigma)
                x = torch.cat((z, x),1)
        
        return x.permute(0, 2, 1).reshape(batch_size, -1)

        

        
        







class RealNVP(nn.Module):
    def __init__(self, latent_dim, num_blocks, scale_net_func, transition_net_func):
        super().__init__()
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim

        self.prior = torch.distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))

        # transition networks
        self.t = nn.ModuleList([transition_net_func() for _ in range(num_blocks)])
        # scale networks
        self.s = nn.ModuleList([scale_net_func() for _ in range(num_blocks)])


    def coupling(self, x, i, forward=True):
        # devide into xa, xb
        (xa, xb) = torch.chunk(x, 2, 1)

        s = self.s[i](xa)
        t = self.t[i](xa)

        if forward: 
            yb = (xb - t) * torch.exp(-s)
        else: 
            yb = torch.exp(s) * xb + t

        # s for log determinant jacobian
        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        # used simple operation : flip
        return x.flip(1) 

    def _f(self, x):
        """
        forward transformation.
        """
        
        # the log determinant jacobian can be calculated by summing the scale of x_a
        ldj = x.new_zeros(x.size(0)) # [B] -> single value
        
        for i in range(self.num_blocks):
            x, s = self.coupling(x, i, True)
            x = self.permute(x)
            ldj = ldj - s.sum(dim=1)
        return x, ldj

    def _inv_f(self, x):
        """
        inverse forward transformation.
        while sampling, it does not calculate the log determinant of jacobian
        """
        for i in reversed(range(self.num_blocks)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, False)
        return x
    
    def forward(self, input):
        # dequantization
        device = input.device
        x = input + (1. - torch.rand(input.shape).to(device))/2.
        x, ldj = self._f(x)
        # when device is cuda:0
        return -(self.prior.log_prob(x.to('cpu')).to(device) + ldj).mean()
    
    def inference(self, batch_size, device):
        x = self.prior.sample((batch_size, )).to(device)
        return self._inv_f(x)
        

