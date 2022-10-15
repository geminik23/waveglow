import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import InvertibleConv1D




class WaveGlowTestCase(unittest.TestCase):
    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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