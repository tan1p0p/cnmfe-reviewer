import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nn.res_block import ResNet
from nn.san_block import SAN

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_cnn_for_trace=True):
        super().__init__()
        self.use_cnn_for_trace = use_cnn_for_trace
        self.conv = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if self.use_cnn_for_trace:
            x = self.conv(x)
        _, (hn, _) = self.lstm(x)
        return F.relu(self.fc(hn.squeeze(0)))

class Model(nn.Module):
    def __init__(self, s_size=(80, 80), t_size=500, s_stage='ResNet', use_cnn_for_trace=True):
        super().__init__()
        if s_stage == 'ResNet':
            self.spatial_stage = ResNet()
        elif s_stage == 'SAN':
            self.spatial_stage = SAN(sa_type=0, layers=(3, 4, 6, 8), kernels=[3, 7, 7, 7])
        self.temporal_stage = CNN_LSTM(t_size, hidden_dim=300, output_dim=t_size, use_cnn_for_trace=use_cnn_for_trace)

        self.out = nn.Sequential(
            nn.Linear(s_size[0] * s_size[1] + t_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        t, s = inputs
        t = self.temporal_stage(t)
        s = torch.flatten(self.spatial_stage(s), 1)
        return self.out(torch.cat((s, t), 1))
