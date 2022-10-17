import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import InvertibleConv1D, WaveNetLike

class WaveGlowTestCase(unittest.TestCase):
    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def test_wnlayer(self):
        b = 16 # batch size
        in_channels = 256
        residual_channels = 256
        skip_channels = 256
        mel_channels = 80
        hidden_channels = 20
        max_dilations = 8
        length = 1000

        layer = WaveNetLike(in_channels, mel_channels, residual_channels, skip_channels, hidden_channels, max_dilations=8)
        self.assertEqual(len(layer.layers), max_dilations+1)

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
        y = iconv(x, reverse=False)
        x_ = iconv(y, reverse=True)[0]

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