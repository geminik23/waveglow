import unittest
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from modules import InvertibleConv1D, WaveNetLike
from model import WaveGlow


class WaveGlowTestCase(unittest.TestCase):
    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_model(self):
        from config import Config
        config = Config()
        n_mels = 80
        win_length = 1024
        hop_length = 256

        n_flows = 12
        n_groups = 8
        n_early_every = 4
        n_early_size = 2
        sigma = 1.0

        # params for wavenetlike layer
        wn_config = {
            'residual_channels': 256,
            'skip_channels': 256,
            'hidden_channels': 256,
            'kernel_size': 3,
            'max_dilation': 8,
        }

        batch_size = 2
        audio_length = 16000
        sample_rate = 16000

        model = WaveGlow(n_mels, n_flows, n_groups, n_early_every, n_early_size, win_length=win_length, hop_length=hop_length, sigma=sigma, wn_config=wn_config)
        x = torch.randn((batch_size, audio_length))
        melspec_op = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=win_length, win_length=win_length, hop_length=hop_length)
        y = melspec_op(x)
        # _loss = model((x, y))

        out = model.inference(y)
        self.assertEqual(out.size(0), batch_size)
        self.assertTrue(out.size(-1) - audio_length < hop_length)

    
    def test_unfold(self):
        x = torch.arange(0, 12).view(2,6)
        for size in range(1, x.size(-1)+1):
            y = x.unfold(1, size, size)
            self.assertEqual(torch.Size([2, x.size(-1)//size, size]), y.shape)


    def test_wnlayer(self):
        b = 16 # batch size
        in_channels = 256
        residual_channels = 256
        skip_channels = 256
        mel_channels = 80
        hidden_channels = 20
        max_dilation = 8
        length = 1000

        layer = WaveNetLike(in_channels, mel_channels, residual_channels, skip_channels, hidden_channels, max_dilation=max_dilation)
        self.assertEqual(len(layer.layers), max_dilation+1)

        dilations = [l.dilation for l in layer.layers]
        self.assertEqual([1,2,4,8,16,32,64,128,256], dilations)

        x = torch.randn(b, in_channels, length)
        cond = torch.randn(b, mel_channels, length)
        y = layer(x, cond)

        target_shape = torch.Size([b, in_channels*2, length])
        self.assertEqual(target_shape, y.shape)



    def test_invertible_1x1_conv(self):
        b = 1 # batch size
        c = 2 # channel size
        l = 10 # length

        device = self.get_device()
        iconv = InvertibleConv1D(c).to(device)
        x = torch.randn((b, c, l), device=device)
        y = iconv(x, reverse=True)
        x_ = iconv(y, reverse=False)[0]

        self.assertTrue((x-x_).sum().item() < 1e-6) 

        pass

    def test_invertible_conv_op(self):
        b = 1 # batch size
        c = 2 # channel size
        l = 10 # length

        x = torch.randn((b, c, l))

        # assume that z is 1d 
        # QR decomposition -> Q is an orthogonal matrix
        conv = nn.Conv1d(c, c, 1, 1, 0)
        q = torch.linalg.qr(torch.randn(c, c))[0].unsqueeze(-1)
        self.assertEqual(conv.weight.data.shape, q.shape)

        W = nn.Parameter(q)
        ##
        # forward 
        y = F.conv1d(x, W)

        ##
        # backward
        x_ = F.conv1d(y, W.squeeze().inverse().unsqueeze(-1))

        self.assertTrue((x-x_).sum().item() < 1e-6) 




if __name__ == '__main__':
    unittest.main(verbosity=2)