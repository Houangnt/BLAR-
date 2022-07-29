import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPIN(nn.Module):
    def __init__(self, nc=1, k=6, device=None):
        super(SPIN, self).__init__()
        self.nc = nc
        self.k = k
        self.device = device

        self.shared_conv, self.spn, self.ain, self.sigmoid, self.pool = get_spn_ain(nc, 2 * self.k + 2)
        self.instance_norm = nn.InstanceNorm2d(nc)
        self.beta = self.get_beta()
        self.initialize_weights()

    def initialize_weights(self):
        for module in [self.shared_conv, self.spn, self.ain]:
            for m in module.modules():
                t = type(m)
                if t is nn.Conv2d:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif t is nn.BatchNorm2d:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    m.eps = 1e-3
                    m.momentum = 0.03
                elif t is nn.Linear:
                    m.weight.data.normal_(0, 0.001)
                    m.bias.data.zero_()
        self.spn.Linear_2.weight.data.fill_(0)
        init = np.array([.0] * self.k + [5.] + [.0] * self.k + [-5.])
        self.spn.Linear_2.bias.data = torch.from_numpy(init).float().view(-1)

    def get_beta(self):
        b1 = []
        b2 = []
        for i in range(1, self.k + 1):
            tmp = round(math.log(1 - 0.5 * i / (self.k + 1)) / math.log(0.5 * i / (self.k + 1)), 2)
            b1.append(tmp)
            b2.append(round(1 / tmp, 2))
        return b1 + [1.] + b2

    def forward(self, x):
        x = x.to(device=self.device)
        feature = self.shared_conv(x)
        sp_output = self.spn(feature).view(x.size(0), 2 * self.k + 2, 1)

        alpha = sp_output[:, 2 * self.k + 1, 0]
        alpha = self.sigmoid(alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        w_sp = sp_output[:, : 2 * self.k + 1, :]
        offsets = self.pool(self.ain(feature))
        offsets = self.sigmoid(offsets)

        offsets = F.upsample(offsets, size=(x.size(2), x.size(3)), mode='bilinear')
        x = .5 * (x + 1) * (1 - alpha) + offsets * alpha
        w_s = w_sp.unsqueeze(-1).unsqueeze(-1)
        powered = torch.stack([x.pow(p) for p in self.beta], dim=1)

        x = (powered * w_s).sum(dim=1)
        x = self.sigmoid(self.instance_norm(x)) * 2 + 1
        return x


def get_spn_ain(input_channel, n_output):
    shared_conv = nn.Sequential(
        nn.Conv2d(input_channel, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(32), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(128), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    spn_output = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(512), nn.ReLU(True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1, -1),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256), nn.ReLU(True)
    )
    spn_output.add_module('Linear_2', nn.Linear(256, n_output))
    ain_output = nn.Sequential(
        nn.Conv2d(128, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        nn.Conv2d(16, input_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    sigmoid = nn.Sigmoid()
    pool = nn.MaxPool2d(2, 2)
    return shared_conv, spn_output, ain_output, sigmoid, pool
