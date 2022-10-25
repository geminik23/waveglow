

from cmath import sqrt


class Config:
    def __init__(self):
        self.n_mels = 80
        self.hop_length = 256
        self.win_length = 1024

        self.n_flows = 12
        self.n_groups = 8
        self.n_early_every = 4
        self.n_early_size = 2

        self.training_std = sqrt(0.5)

        # params for wavenetlike layer
        self.wn_config = {
            'residual_channels': 256,
            'skip_channels': 256,
            'hidden_channels': 256,
            'kernel_size': 3,
            'max_dilation': 8,
        }