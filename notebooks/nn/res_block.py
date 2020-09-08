import torch.nn as nn
import torch.nn.functional as F

from nn.san_block import conv1x1

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def deconv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        channel = channel_out // 4

        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel_out, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channel_out)

        self.shortcut = self._shortcut(channel_in, channel_out)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return F.relu(x + self.shortcut(x)) # skip connection

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=1)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 32
        self.conv_in, self.bn_in = conv1x1(1, c), nn.BatchNorm2d(c)
        self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.residual0 = ResBlock(c, c)

        c *= 2
        self.conv1, self.bn1 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.residual1 = ResBlock(c, c)

        c *= 2
        self.conv2, self.bn2 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.residual2 = ResBlock(c, c)

        c *= 2
        self.conv3, self.bn3 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.residual3 = ResBlock(c, c)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.relu(self.bn0(self.residual0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.residual1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.residual2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.residual3(self.conv3(self.pool(x)))))
        return x
