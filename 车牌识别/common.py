# -*- coding: utf-8 -*-
"""
author:LTH
data:
"""
from torch import nn
from torch.nn import functional as F


class HSwish(nn.Module):

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'hard_swish':
            self.act = HSwish()
        elif act is None:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBNACTWithPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=None):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
        # self.pool = SoftPooling2D(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2,
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act is None:
            self.act = None
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, name, if_first=False):
        super().__init__()
        assert name is not None, 'shortcut must have name'

        self.name = name
        if in_channels != out_channels or stride[0] != 1:
            if if_first:
                self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                      padding=0, groups=1, act=None)
            else:
                self.conv = ConvBNACTWithPool(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                              stride=stride, groups=1, act=None)
        elif if_first:
            self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                  padding=0, groups=1, act=None)
        else:
            self.conv = None

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super().__init__()
        assert name is not None, 'block must have name'
        self.name = name

        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                 name=f'{name}_branch1', if_first=if_first, )
        self.relu = nn.ReLU()
        self.output_channels = out_channels

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super().__init__()
        assert name is not None, 'bottleneck must have name'
        self.name = name
        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv2 = ConvBNACT(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, stride=1,
                               padding=0, groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * 4, stride=stride,
                                 if_first=if_first, name=f'{name}_branch1')
        self.relu = nn.ReLU()
        self.output_channels = out_channels * 4

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class HardSigmoid(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type

    def forward(self, x):
        if self.type == 'paddle':
            x = (1.2 * x).add_(3.).clamp_(0., 6.).div_(6.)
        else:
            x = F.relu6(x + 3, inplace=True) / 6
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hsigmoid_type='others', ratio=4):
        super().__init__()
        num_mid_filter = out_channels // ratio
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_mid_filter, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter, kernel_size=1, out_channels=out_channels, bias=True)
        self.relu2 = HardSigmoid(hsigmoid_type)

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn


class ResidualUnit(nn.Module):
    def __init__(self, num_in_filter, num_mid_filter, num_out_filter, stride, kernel_size, act=None, use_se=False):
        super().__init__()
        self.expand_conv = ConvBNACT(in_channels=num_in_filter, out_channels=num_mid_filter, kernel_size=1, stride=1,
                                     padding=0, act=act)

        self.bottleneck_conv = ConvBNACT(in_channels=num_mid_filter, out_channels=num_mid_filter,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=int((kernel_size - 1) // 2), act=act, groups=num_mid_filter)
        if use_se:
            self.se = SEBlock(in_channels=num_mid_filter, out_channels=num_mid_filter, hsigmoid_type='paddle')
        else:
            self.se = None

        self.linear_conv = ConvBNACT(in_channels=num_mid_filter, out_channels=num_out_filter, kernel_size=1, stride=1,
                                     padding=0)
        self.not_add = num_in_filter != num_out_filter or stride != 1

    def forward(self, x):
        y = self.expand_conv(x)
        y = self.bottleneck_conv(y)
        if self.se is not None:
            y = self.se(y)
        y = self.linear_conv(y)
        if not self.not_add:
            y = x + y
        return y


class DecoderWithRNN(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        rnn_hidden_size = kwargs.get('hidden_size', 96)
        self.out_channels = rnn_hidden_size * 2
        self.layers = 2
        self.lstm = nn.LSTM(in_channels, rnn_hidden_size, bidirectional=True, batch_first=True, num_layers=self.layers)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        return x


class Reshape(nn.Module):
    def __init__(self, in_channels=256, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = x.permute((0, 2, 1))  # (NTC)(batch, width, channel)s
        return x
