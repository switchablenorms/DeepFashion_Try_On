### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import os
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from grid_sample import grid_sample
from torch.autograd import Variable
from tps_grid_gen import TPSGridGen
import ipdb

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, L=1, S=1, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, L, S, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer)
    else:
        raise ('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_Unet(input_nc, gpu_ids=[]):
    netG = Unet(input_nc)
    netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_UnetMask(input_nc, gpu_ids=[]):
    netG = UnetMask(input_nc,output_nc=4)
    netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_Refine(input_nc, output_nc, gpu_ids=[]):
    netG = Refine(input_nc, output_nc)
    netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def define_VAE(input_nc, gpu_ids=[]):
    netVAE = VAE(19, 32, 32, 1024)
    print(netVAE)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netVAE.cuda(gpu_ids[0])
    return netVAE


def define_B(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netB = BlendGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    print(netB)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netB.cuda(gpu_ids[0])
    netB.apply(weights_init)
    return netB


def define_partial_enc(input_nc, gpu_ids=[]):
    net = PartialConvEncoder(input_nc)
    print(net)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net


def define_conv_enc(input_nc, gpu_ids=[]):
    net = ConvEncoder(input_nc)
    print(net)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net


def define_AttG(output_nc, gpu_ids=[]):
    net = AttGenerator(output_nc)
    print(net)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLossWarp(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLossWarp, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        loss += self.weights[4] * self.criterion(x_vgg[4], y_vgg[4].detach())
        return loss


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def warp(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        loss += self.weights[4] * self.criterion(x_vgg[4], y_vgg[4].detach())
        return loss


class StyleLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(StyleLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            N, C, H, W = x_vgg[i].shape
            for n in range(N):
                phi_x = x_vgg[i][n]
                phi_y = y_vgg[i][n]
                phi_x = phi_x.reshape(C, H * W)
                phi_y = phi_y.reshape(C, H * W)
                G_x = torch.matmul(phi_x, phi_x.t()) / (C * H * W)
                G_y = torch.matmul(phi_y, phi_y.t()) / (C * H * W)
                loss += torch.sqrt(torch.mean((G_x - G_y) ** 2)) * self.weights[i]
        return loss


##############################################################################
# Generator
##############################################################################

class PartialConvEncoder(nn.Module):
    def __init__(self, input_nc, ngf=32, norm_layer=nn.BatchNorm2d):
        super(PartialConvEncoder, self).__init__()
        activation = nn.ReLU(True)
        self.pad1 = nn.ReflectionPad2d(3)
        self.partial_conv1 = PartialConv(input_nc, ngf, kernel_size=7)
        self.norm_layer1 = norm_layer(ngf)
        self.activation = activation
        ##down sample
        mult = 2 ** 0
        self.down1 = PartialConv(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
        self.norm_layer2 = norm_layer(ngf * mult * 2)
        mult = 2 ** 1
        self.down2 = PartialConv(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
        self.norm_layer3 = norm_layer(ngf * mult * 2)

        mult = 2 ** 2
        self.down3 = PartialConv(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
        self.norm_layer4 = norm_layer(ngf * mult * 2)

        mult = 2 ** 3
        self.down4 = PartialConv(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
        self.norm_layer5 = norm_layer(ngf * mult * 2)

    def forward(self, input, mask):
        input = self.pad1(input)
        mask = self.pad1(mask)
        input, mask = self.partial_conv1(input, mask)
        input = self.norm_layer1(input)
        input = self.activation(input)

        input, mask = self.down1(input, mask)
        input = self.norm_layer2(input)
        input = self.activation(input)
        input, mask = self.down2(input, mask)
        input = self.norm_layer3(input)
        input = self.activation(input)
        input, mask = self.down3(input, mask)
        input = self.norm_layer4(input)
        input = self.activation(input)
        input, mask = self.down4(input, mask)
        input = self.norm_layer5(input)
        input = self.activation(input)
        return input


class ConvEncoder(nn.Module):
    def __init__(self, input_nc, ngf=32, n_downsampling=4, n_blocks=4, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        super(ConvEncoder, self).__init__()
        activation = nn.ReLU(True)
        # print("input_nc",input_nc)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            stride = 2

            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=stride, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class AttGenerator(nn.Module):
    def __init__(self, output_nc, ngf=32, n_blocks=4, n_downsampling=4, padding_type='reflect'):
        super(AttGenerator, self).__init__()
        mult = 2 ** n_downsampling
        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult * 2, norm_type='in', padding_type=padding_type)]

        self.model = nn.Sequential(*model)
        self.upsampling = []
        self.out_channels = []
        self.AttNorm = []
        ##upsampling
        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU(True)

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            up_module = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2) * 2, kernel_size=3, stride=2, padding=1,
                                            output_padding=1),
                         norm_layer(int(ngf * mult / 2) * 2), activation
                         ]
            up_module = nn.Sequential(*up_module)
            self.upsampling += [up_module]
            self.out_channels += [int(ngf * mult / 2) * 2]
        self.upsampling = nn.Sequential(*self.upsampling)

        #
        self.AttNorm += [AttentionNorm(5, self.out_channels[0], 2, 4)]
        self.AttNorm += [AttentionNorm(5, self.out_channels[1], 2, 2)]
        self.AttNorm += [AttentionNorm(5, self.out_channels[2], 1, 2)]
        self.AttNorm += [AttentionNorm(5, self.out_channels[3], 1, 1)]
        self.AttNorm = nn.Sequential(*self.AttNorm)
        self.last_conv = [nn.ReflectionPad2d(3), nn.Conv2d(ngf * 2, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.last_conv = nn.Sequential(*self.last_conv)

    def forward(self, input, unattended):
        up = self.model(unattended)
        for i in range(4):
            # print(i)
            up = self.upsampling[i](up)
            if i == 3:
                break;
            up = self.AttNorm[i](input, up)
        return self.last_conv(up)


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class AttentionNorm(nn.Module):
    def __init__(self, ref_channels, out_channels, first_rate, second_rate):
        super(AttentionNorm, self).__init__()
        self.first = first_rate
        self.second = second_rate
        mid_channels = int(out_channels / 2)
        self.conv_1time_f = nn.Conv2d(ref_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2times_f = nn.Conv2d(ref_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.conv_4times_f = nn.Conv2d(ref_channels, mid_channels, kernel_size=3, stride=4, padding=1)

        self.conv_1time_s = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2times_s = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv_4times_s = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=4, padding=1)

        self.conv_1time_m = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2times_m = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv_4times_m = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=4, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input, unattended):
        # attention weights
        # print(input.shape,unattended.shape)
        if self.first == 1:
            input = self.conv_1time_f(input)
        elif self.first == 2:
            input = self.conv_2times_f(input)
        elif self.first == 4:
            input = self.conv_4times_f(input)
        mask = None
        if self.second == 1:
            bias = self.conv_1time_s(input)
            mask = self.conv_1time_m(input)
        elif self.second == 2:
            bias = self.conv_2times_s(input)
            mask = self.conv_2times_m(input)
        elif self.second == 4:
            bias = self.conv_4times_s(input)
            mask = self.conv_4times_m(input)
        mask = torch.sigmoid(mask)
        attended = self.norm(unattended)
        # print(attended.shape,mask.shape,bias.shape)
        attended = attended * mask + bias
        attended = torch.relu(attended)
        attended = self.conv(attended)
        output = attended + unattended
        return output
class UnetMask(nn.Module):
    def __init__(self, input_nc, output_nc=3):
        super(UnetMask, self).__init__()
        self.stn = STNNet()
        nl = nn.InstanceNorm2d
        self.conv1 = nn.Sequential(*[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU()])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Sequential(*[nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU(),
                                     nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU()])
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512),
              nn.ReLU()])

        self.conv6 = nn.Sequential(*[nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.up7 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256),
              nn.ReLU()])
        self.conv7 = nn.Sequential(*[nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])

        self.up8 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128),
              nn.ReLU()])

        self.conv8 = nn.Sequential(*[nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])

        self.up9 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64),
              nn.ReLU()])

        self.conv9 = nn.Sequential(*[nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1)
                                     ])

    def forward(self, input, refer, mask):
        input, warped_mask,rx,ry,cx,cy = self.stn(input, torch.cat([mask, refer, input], 1), mask)
        #ipdb.set_trace()# print(input.shape)

        conv1 = self.conv1(torch.cat([refer.detach(), input.detach()], 1))
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        conv6 = self.conv6(torch.cat([drop4, up6], 1))

        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], 1))

        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], 1))

        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], 1))
        return conv9, input, warped_mask,rx,ry,cx,cy

class Unet(nn.Module):
    def __init__(self, input_nc, output_nc=3):
        super(Unet, self).__init__()
        self.stn = STNNet()
        nl = nn.InstanceNorm2d
        self.conv1 = nn.Sequential(*[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU()])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Sequential(*[nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU(),
                                     nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU()])
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512),
              nn.ReLU()])

        self.conv6 = nn.Sequential(*[nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.up7 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256),
              nn.ReLU()])
        self.conv7 = nn.Sequential(*[nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])

        self.up8 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128),
              nn.ReLU()])

        self.conv8 = nn.Sequential(*[nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])

        self.up9 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64),
              nn.ReLU()])

        self.conv9 = nn.Sequential(*[nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1)
                                     ])

    def forward(self, input, refer, mask):
        input, warped_mask,rx,ry,cx,cy = self.stn(input, torch.cat([mask, refer, input], 1), mask)
        #ipdb.set_trace()
        conv1 = self.conv1(torch.cat([refer.detach(), input.detach()], 1))
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        conv6 = self.conv6(torch.cat([drop4, up6], 1))

        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], 1))

        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], 1))

        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], 1))
        return conv9, input, warped_mask,rx,ry,cx,cy

    def refine(self, input):
        conv1 = self.conv1(input)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        conv6 = self.conv6(torch.cat([drop4, up6], 1))

        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], 1))

        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], 1))

        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], 1))
        return conv9


class Refine(nn.Module):
    def __init__(self, input_nc, output_nc=3):
        super(Refine, self).__init__()
        nl = nn.InstanceNorm2d
        self.conv1 = nn.Sequential(*[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU()])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Sequential(*[nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU(),
                                     nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU()])
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512),
              nn.ReLU()])

        self.conv6 = nn.Sequential(*[nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.up7 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256),
              nn.ReLU()])
        self.conv7 = nn.Sequential(*[nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])

        self.up8 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128),
              nn.ReLU()])

        self.conv8 = nn.Sequential(*[nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])

        self.up9 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64),
              nn.ReLU()])

        self.conv9 = nn.Sequential(*[nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1)
                                     ])

    def refine(self, input):
        conv1 = self.conv1(input)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        conv6 = self.conv6(torch.cat([drop4, up6], 1))

        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], 1))

        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], 1))

        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], 1))
        return conv9


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, L, S, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type='adain', padding_type=padding_type)]
        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

        # style encoder
        self.enc_style = StyleEncoder(5, S, 16, self.get_num_adain_params(self.model), norm='none', activ='relu',
                                      pad_type='reflect')
        # label encoder
        self.enc_label = LabelEncoder(5, L, 16, 64, norm='none', activ='relu', pad_type='reflect')

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def forward(self, input, input_ref, image_ref):
        fea1, fea2 = self.enc_label(input_ref)
        adain_params = self.enc_style((image_ref, fea1, fea2))
        self.assign_adain_params(adain_params, self.model)
        return self.model(input)


class BlendGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=3, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(BlendGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_type='in', padding_type=padding_type)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, input1, input2):
        m = self.model(torch.cat([input1, input2], 1))
        return input1 * m + input2 * (1 - m), m

    # Define the Multiscale Discriminator.


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Define the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=False)
        vgg.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + "/vgg19-dcbb9e9d.pth"))
        vgg_pretrained_features = vgg.features
        self.vgg = vgg
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    def extract(self, x):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        return x


# Define the MaskVAE
class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()
        # self.cuda = True
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.e3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.e4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.e5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf * 16)

        self.e6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(ndf * 32)

        self.e7 = nn.Conv2d(ndf * 32, ndf * 64, 4, 2, 1)
        self.bn7 = nn.BatchNorm2d(ndf * 64)

        self.fc1 = nn.Linear(ndf * 64 * 4 * 4, latent_variable_size)
        self.fc2 = nn.Linear(ndf * 64 * 4 * 4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf * 64 * 4 * 4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf * 64, ngf * 32, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf * 32, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf * 32, ngf * 16, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf * 16, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf * 16, ngf * 8, 3, 1)
        self.bn10 = nn.BatchNorm2d(ngf * 8, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf * 8, ngf * 4, 3, 1)
        self.bn11 = nn.BatchNorm2d(ngf * 4, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf * 4, ngf * 2, 3, 1)
        self.bn12 = nn.BatchNorm2d(ngf * 2, 1.e-3)

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd6 = nn.ReplicationPad2d(1)
        self.d7 = nn.Conv2d(ngf * 2, ngf, 3, 1)
        self.bn13 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up7 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd7 = nn.ReplicationPad2d(1)
        self.d8 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h6 = self.leakyrelu(self.bn6(self.e6(h5)))
        h7 = self.leakyrelu(self.bn7(self.e7(h6)))
        h7 = h7.view(-1, self.ndf * 64 * 4 * 4)
        return self.fc1(h7), self.fc2(h7)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # if self.cuda:
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        # else:
        #    eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf * 64, 4, 4)
        h2 = self.leakyrelu(self.bn8(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn9(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn10(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn11(self.d5(self.pd4(self.up4(h4)))))
        h6 = self.leakyrelu(self.bn12(self.d6(self.pd5(self.up5(h5)))))
        h7 = self.leakyrelu(self.bn13(self.d7(self.pd6(self.up6(h6)))))
        return self.d8(self.pd7(self.up7(h7)))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar.mul(0.5).exp_()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)

        return res, x, mu, logvar


# style encode part
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model_middle = []
        self.model_last = []
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model_middle += [ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model_last += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model_last += [nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.model_middle = nn.Sequential(*self.model_middle)
        self.model_last = nn.Sequential(*self.model_last)

        self.output_dim = dim

        self.sft1 = SFTLayer()
        self.sft2 = SFTLayer()

    def forward(self, x):
        fea = self.model(x[0])
        fea = self.sft1((fea, x[1]))
        fea = self.model_middle(fea)
        fea = self.sft2((fea, x[2]))
        return self.model_last(fea)


# label encode part
class LabelEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(LabelEncoder, self).__init__()
        self.model = []
        self.model_last = [nn.ReLU()]
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim *= 2
        self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation='none', pad_type=pad_type)]
        dim *= 2
        for i in range(n_downsample - 3):
            self.model_last += [ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model_last += [ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.model_last = nn.Sequential(*self.model_last)
        self.output_dim = dim

    def forward(self, x):
        fea = self.model(x)
        return fea, self.model_last(fea)


# Define the basic block
class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ConvBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_type, padding_type, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_type, padding_type, use_dropout)

    def build_conv_block(self, dim, norm_type, padding_type, use_dropout):
        conv_block = []
        conv_block += [ConvBlock(dim, dim, 3, 1, 1, norm=norm_type, activation='relu', pad_type=padding_type)]
        conv_block += [ConvBlock(dim, dim, 3, 1, 1, norm=norm_type, activation='none', pad_type=padding_type)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv1 = nn.Conv2d(64, 64, 1)
        self.SFT_scale_conv2 = nn.Conv2d(64, 64, 1)
        self.SFT_shift_conv1 = nn.Conv2d(64, 64, 1)
        self.SFT_shift_conv2 = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv2(F.leaky_relu(self.SFT_scale_conv1(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv2(F.leaky_relu(self.SFT_shift_conv1(x[1]), 0.1, inplace=True))
        return x[0] * scale + shift


class ConvBlock_SFT(nn.Module):
    def __init__(self, dim, norm_type, padding_type, use_dropout=False):
        super(ResnetBlock_SFT, self).__init__()
        self.sft1 = SFTLayer()
        self.conv1 = ConvBlock(dim, dim, 4, 2, 1, norm=norm_type, activation='none', pad_type=padding_type)

    def forward(self, x):
        fea = self.sft1((x[0], x[1]))
        fea = F.relu(self.conv1(fea), inplace=True)
        return (x[0] + fea, x[1])


class ConvBlock_SFT_last(nn.Module):
    def __init__(self, dim, norm_type, padding_type, use_dropout=False):
        super(ResnetBlock_SFT_last, self).__init__()
        self.sft1 = SFTLayer()
        self.conv1 = ConvBlock(dim, dim, 4, 2, 1, norm=norm_type, activation='none', pad_type=padding_type)

    def forward(self, x):
        fea = self.sft1((x[0], x[1]))
        fea = F.relu(self.conv1(fea), inplace=True)
        return x[0] + fea


# Definition of normalization layer
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


### STN TPS

class CNN(nn.Module):
    def __init__(self, num_output, input_nc=5, ngf=8, n_layers=5, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(CNN, self).__init__()
        downconv = nn.Conv2d(5, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2 ** i * ngf if 2 ** i * ngf < 1024 else 1024
            out_ngf = 2 ** (i + 1) * ngf if 2 ** i * ngf < 1024 else 1024
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, norm_layer(out_ngf), nn.ReLU(True)]
        model += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  norm_layer(64), nn.ReLU(True)]
        model += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  norm_layer(64), nn.ReLU(True)]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_output)
    def forward(self, x):
        x = self.model(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


class ClsNet(nn.Module):

    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(10)

    def forward(self, x):
        return F.log_softmax(self.cnn(x))


class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        #ipdb.set_trace()
        coor=points.view(batch_size, -1, 2)
        row=self.get_row(coor,5)
        col=self.get_col(coor,5)
        rx,ry,cx,cy=torch.tensor(0.08).cuda(),torch.tensor(0.08).cuda()\
            ,torch.tensor(0.08).cuda(),torch.tensor(0.08).cuda()
        row_x,row_y=row[:,:,0],row[:,:,1]
        col_x,col_y=col[:,:,0],col[:,:,1]
        rx_loss=torch.max(rx,row_x).mean()
        ry_loss=torch.max(ry,row_y).mean()
        cx_loss=torch.max(cx,col_x).mean()
        cy_loss=torch.max(cy,col_y).mean()


        return  coor,rx_loss,ry_loss,cx_loss,cy_loss

    def get_row(self,coor,num):
        sec_dic=[]
        for j in range(num):
            sum=0
            buffer=0
            flag=False
            max=-1
            for i in range(num-1):
                differ=(coor[:,j*num+i+1,:]-coor[:,j*num+i,:])**2
                if not flag:
                    second_dif=0
                    flag=True
                else:
                    second_dif=torch.abs(differ-buffer)
                    sec_dic.append(second_dif)

                buffer=differ
                sum+=second_dif
        return torch.stack(sec_dic,dim=1)

    def get_col(self,coor,num):
        sec_dic=[]
        for i in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for j in range(num - 1):
                differ = (coor[:, (j+1) * num + i , :] - coor[:, j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ-buffer)
                    sec_dic.append(second_dif)
                buffer = differ
                sum += second_dif
        return torch.stack(sec_dic,dim=1)

class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)


class STNNet(nn.Module):

    def __init__(self):
        super(STNNet, self).__init__()
        range = 0.9
        r1 = range
        r2 = range
        grid_size_h = 5
        grid_size_w = 5

        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_size_h - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_size_w - 1)),
        )))
        #ipdb.set_trace()
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)
        # self.get_row(target_control_points,5)
        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }['bounded_stn']
        self.loc_net = GridLocNet(grid_size_h, grid_size_w, target_control_points)

        self.tps = TPSGridGen(256, 192, target_control_points)

    def get_row(self, coor, num):
        for j in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for i in range(num - 1):
                differ = (coor[j * num + i + 1, :] - coor[j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ - buffer)

                buffer = differ
                sum += second_dif
            print(sum / num)
    def get_col(self,coor,num):
        for i in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for j in range(num - 1):
                differ = (coor[ (j + 1) * num + i, :] - coor[j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ-buffer)

                buffer = differ
                sum += second_dif
            print(sum)
    def forward(self, x, reference, mask):
        batch_size = x.size(0)
        source_control_points,rx,ry,cx,cy = self.loc_net(reference)
        source_control_points=(source_control_points)
        # print('control points',source_control_points.shape)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, 256, 192, 2)
        # print('grid size',grid.shape)
        transformed_x = grid_sample(x, grid, canvas=0)
        warped_mask = grid_sample(mask, grid, canvas=0)
        return transformed_x, warped_mask,rx,ry,cx,cy
