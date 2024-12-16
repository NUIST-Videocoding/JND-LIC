import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.uniform import Uniform
from compressai.zoo import bmshj2018_hyperprior, bmshj2018_factorized
from .basic_module import Non_local_Block, ResBlock, GDN,  JND_FTM, EfficientAttention
from .jnd_context_atten import P_Model
from .factorized_entropy_model import Entropy_bottleneck
from .gaussian_entropy_model import Distribution_for_entropy


class Enc(nn.Module):
    def __init__(self, num_features, N1, N2, M, M1):
        # input_features = 3, N1 = 192, N2 = 128, M = 192, M1 = 96
        super(Enc, self).__init__()
        self.N1 = int(N1)
        self.N2 = int(N2)
        self.M = int(M)
        self.M1 = int(M1)
        self.n_features = int(num_features)

        self.conv1 = nn.Conv2d(self.n_features, self.M1, 5, 1, 2)
        self.trunk1 = nn.Sequential(ResBlock(self.M1, self.M1, 3, 1, 1), ResBlock(
            self.M1, self.M1, 3, 1, 1), nn.Conv2d(self.M1, 2 * self.M1, 5, 2, 2))
        self.atten1 = JND_FTM(self.M)
        self.down1 = nn.Conv2d(2 * self.M1, self.M, 5, 2, 2)
        self.trunk2 = nn.Sequential(ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1))

        self.mask1 = nn.Sequential(EfficientAttention(2 * self.M1, self.M1, head_count=1, value_channels=self.M1),
                                   ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                   ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1), 
                                   ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                   nn.Conv2d(2 * self.M1, 2 * self.M1, 1, 1, 0))
        
        self.trunk3 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 5, 2, 2))
        self.atten3 = JND_FTM(self.M)
        self.trunk4 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 5, 2, 2))

        self.trunk5 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1))
        self.mask2 = nn.Sequential(Non_local_Block(self.M, self.M // 2),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 1, 1, 0))

        # hyper

        self.trunk6 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    nn.Conv2d(self.M, self.M, 5, 2, 2))
        self.trunk7 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    nn.Conv2d(self.M, self.M, 5, 2, 2))

        self.trunk8 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1))
        self.mask3 = nn.Sequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 1, 1, 0))
        self.conv2 = nn.Conv2d(self.M, self.N2, 3, 1, 1)

    def forward(self, x, jnd):
        x1 = self.conv1(x)
        jnd1 = self.conv1(jnd)
        x2 = self.trunk1(x1)
        jnd2 = self.trunk1(jnd1)
        x2, jnd2 = self.atten1(x2, jnd2)
        x3 = self.trunk2(x2) * f.sigmoid(self.mask1(x2)) + x2
        jnd3 = self.trunk2(jnd2) * f.sigmoid(self.mask1(jnd2)) + jnd2
        x3 = self.down1(x3)
        jnd3 = self.down1(jnd3)
        x4 = self.trunk3(x3)
        jnd4 = self.trunk3(jnd3)
        x4, jnd4 = self.atten3(x4, jnd4)
        x5 = self.trunk4(x4)
        jnd5 = self.trunk4(jnd4)
        # x5 = self.atten4(x5, jnd5)
        x6 = self.trunk5(x5) * f.sigmoid(self.mask2(x5)) + x5
        jnd6 = self.trunk5(jnd5) * f.sigmoid(self.mask2(jnd5)) + jnd5

        # hyper
        x7 = self.trunk6(x6)
        x8 = self.trunk7(x7)
        x9 = self.trunk8(x8) * f.sigmoid(self.mask3(x8)) + x8
        x10 = self.conv2(x9)

        return x6, x10, jnd6


class Hyper_Dec(nn.Module):
    def __init__(self, N2, M):
        super(Hyper_Dec, self).__init__()

        self.N2 = N2
        self.M = M
        self.conv1 = nn.Conv2d(self.N2, M, 3, 1, 1)
        self.trunk1 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1))
        self.mask1 = nn.Sequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 1, 1, 0))

        self.trunk2 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    nn.ConvTranspose2d(M, M, 5, 2, 2, 1))
        self.trunk3 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    nn.ConvTranspose2d(M, M, 5, 2, 2, 1))

    def forward(self, xq2):
        x1 = self.conv1(xq2)
        x2 = self.trunk1(x1) * f.sigmoid(self.mask1(x1)) + x1
        x3 = self.trunk2(x2)
        x4 = self.trunk3(x3)

        return x4


class Dec(nn.Module):
    def __init__(self, input_features, N1, M, M1):
        super(Dec, self).__init__()

        self.N1 = N1
        self.M = M
        self.M1 = M1
        self.input = input_features
        self.atten3 = JND_FTM(self.M)
        self.atten4 = JND_FTM(self.M)
        self.trunk1 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1))
        self.mask1 = nn.Sequential(Non_local_Block(self.M, self.M // 2),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 1, 1, 0))

        self.up1 = nn.ConvTranspose2d(M, M, 5, 2, 2, 1)
        self.trunk2 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), nn.ConvTranspose2d(M, M, 5, 2, 2, 1))
        self.trunk3 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), nn.ConvTranspose2d(M, 2 * self.M1, 5, 2, 2, 1))

        self.trunk4 = nn.Sequential(ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1))
        self.mask2 = nn.Sequential(EfficientAttention(2 * self.M1, self.M1, head_count=1, value_channels=self.M1),
                                   ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                   ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                   ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                   nn.Conv2d(2 * self.M1, 2 * self.M1, 1, 1, 0))

        self.trunk5 = nn.Sequential(nn.ConvTranspose2d(2 * M1, M1, 5, 2, 2, 1), ResBlock(self.M1, self.M1, 3, 1, 1),
                                    ResBlock(self.M1, self.M1, 3, 1, 1),
                                    ResBlock(self.M1, self.M1, 3, 1, 1))

        self.conv1 = nn.Conv2d(self.M1, self.input, 5, 1, 2)

    def forward(self, x, jnd):
        x1 = self.trunk1(x) * f.sigmoid(self.mask1(x)) + x
        jnd1 = self.trunk1(jnd) * f.sigmoid(self.mask1(jnd)) + jnd
        x1 = self.up1(x1)
        jnd1 = self.up1(jnd1)
        x2 = self.trunk2(x1)
        jnd2 = self.trunk2(jnd1)
        x2, jnd2 = self.atten3(x2, jnd2)
        x3 = self.trunk3(x2)
        jnd3 = self.trunk3(jnd2)
        x4 = self.trunk4(x3) * f.sigmoid(self.mask2(x3)) + x3
        jnd4 = self.trunk4(jnd3) * f.sigmoid(self.mask2(jnd3)) + jnd3
        x4, jnd4 = self.atten4(x4, jnd4)
        # print (x4.size())
        x5 = self.trunk5(x4)
        output = self.conv1(x5)
        return output






class Image_coding(nn.Module):
    def __init__(self, input_features, N1, N2, M, M1, patch_size):
        # input_features = 3, N1 = 192, N2 = 128, M = 192, M1 = 96
        super(Image_coding, self).__init__()
        self.N1 = N1
        self.encoder = Enc(input_features, N1, N2, M, M1)
        self.factorized_entropy_func = Entropy_bottleneck(N2)
        self.hyper_dec = Hyper_Dec(N2, M)
        self.p = P_Model(M)
        self.gaussin_entropy_func = Distribution_for_entropy()
        self.decoder = Dec(input_features, N1, M, M1)
        self.jnd_compress = bmshj2018_hyperprior(quality=2, pretrained=True)
        self.sig = nn.Sigmoid()
        self.maxpool = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=2, stride=2)
        self.unpool = nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=2, stride=2)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def quant(self, inputs, jnd):
        noise = np.random.uniform(-0.5, 0.5, inputs.size())
        noise = torch.Tensor(noise).cuda()
        noise = torch.mul(noise, jnd)
        return inputs + noise

    def quant_test(self, outputs, jnd):
        jnd_for_quantize = jnd
        outputs = torch.mul(outputs, 1 / jnd_for_quantize)
        outputs = torch.round(outputs)
        outputs = torch.mul(outputs, jnd_for_quantize)
        return outputs

    def ste_round(self, inputs, jnd):
        outputs = (torch.round(inputs/jnd)*jnd - inputs).detach() + inputs
        return outputs

    def forward(self, x, jnd, if_training=1):
        x1, x2, jndfeature = self.encoder(x, jnd)
        xq2, xp2 = self.factorized_entropy_func(x2, if_training)
        jnd_down_feature = self.maxpool(jndfeature)
        (b, c, h, w) = jnd_down_feature.shape

        height = int(math.sqrt(c / 3) * h)
        weight = int(math.sqrt(c / 3) * w)

        jnd_feature = jnd_down_feature.reshape(b, 3, height, weight)
        jnd_out = self.jnd_compress(jnd_feature)
        jnd_for_quantize = jnd_out["x_hat"].reshape(b, c, h, w)
        jnd_for_quantize = self.unpool(jnd_for_quantize)
        jnd_for_quantize = torch.clamp(jnd_for_quantize, min=1e-6)
        # jnd_for_quantize = self.sig(jnd_for_quantize)*5
        x3 = self.hyper_dec(xq2)
        hyper_dec = self.p(x3)

        # xq1 = self.ste_round(x1, jnd_for_quantize)
        if if_training == 0:
            xq1 = self.quant(x1, jnd_for_quantize)
        else:
            xq1 = self.quant_test(x1, jnd_for_quantize)
        output = self.decoder(xq1, jnd_for_quantize)

        return [output, xp2, xq1, hyper_dec, jnd_for_quantize, jnd_out]


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = np.random.uniform(-1, 1)
        # b = 0
        uniform_distribution = Uniform(-0.5 * torch.ones(x.size())
                                       * (2 ** b), 0.5 * torch.ones(x.size()) * (2 ** b)).sample().cuda()
        return torch.round(x + uniform_distribution) - uniform_distribution

    @staticmethod
    def backward(ctx, g):
        return g