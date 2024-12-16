import torch
import torch.nn as nn
import torch.nn.functional as f
from .GDN_transform import GDN


class ResGDN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, inv=False):
        super(ResGDN, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.inv = bool(inv)
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.ac1 = GDN(self.in_ch, self.inv)
        self.ac2 = GDN(self.in_ch, self.inv)

    def forward(self, x):
        x1 = self.ac1(self.conv1(x))
        x2 = self.conv2(x1)
        out = self.ac2(x + x2)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)

    def forward(self, x):
        x1 = self.conv2(f.relu(self.conv1(x)))
        out = x+x1
        return out

# here use embedded gaussian

class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = f.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class Non_local_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Non_local_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.theta = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.W = nn.Conv2d(self.out_channel, self.in_channel, 1, 1, 0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        # x_size: (b c h w)

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.out_channel, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.out_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.out_channel, -1)

        f1 = torch.matmul(theta_x, phi_x)
        f_div_C = f.softmax(f1, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.out_channel, *x.size()[2:])
        W_y = self.W(y)
        z = W_y+x

        return z


class JND_FTM(nn.Module):
    def __init__(self, channel1):
        super(JND_FTM, self).__init__()
        self.filter_gate = nn.Conv2d(channel1, channel1, kernel_size=3, padding=1)
        self.W_r = nn.Conv2d(2*channel1, channel1, kernel_size=3, padding=1)
        self.W_s = nn.Conv2d(2*channel1, channel1, kernel_size=3, padding=1)
        self.W_c = nn.Conv2d(2*channel1, channel1, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x1_ = self.filter_gate(x1)
        x2_ = self.filter_gate(x2)
        s = torch.sigmoid(self.W_s(torch.cat((x1_, x2_), 1)))
        s_ = s * x1_
        c = torch.tanh(self.W_c(torch.cat((s_, x2_), 1)))
        r = torch.sigmoid(self.W_r(torch.cat((x1_, x2_), 1)))
        x1_ = -(1 - r) * x1_ + r * c
        return x1_, x2_