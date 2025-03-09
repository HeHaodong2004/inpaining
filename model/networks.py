import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils

from utils.tools import extract_image_patches, flow_to_image, \
    reduce_mean, reduce_sum, default_loader, same_padding


class Generator(nn.Module):
    def __init__(self, config, use_cuda, device_ids, dropout_p=0.05):
        super(Generator, self).__init__()
        if 'netG' in config:
            netG_config = config['netG']
        else:
            netG_config = config
        self.input_dim = netG_config['input_dim']
        self.past_channels = netG_config.get('past_channels', 5)
        self.cnum = netG_config.get('ngf', config.get('ngf', 32))
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        total_in_dim = self.input_dim + self.past_channels

        self.coarse_generator = CoarseGenerator(total_in_dim, self.cnum, use_cuda, device_ids, dropout_p=dropout_p)
        self.fine_generator = FineGenerator(total_in_dim, self.cnum, use_cuda, device_ids, dropout_p=dropout_p)

    def forward(self, x, mask, past=None, dropout=True):

        if dropout:
            self.coarse_generator.train()
            self.fine_generator.train()
        else:
            self.coarse_generator.eval()
            self.fine_generator.eval()
        if past is None:
            B, _, H, W = x.shape
            past = torch.zeros(B, self.past_channels, H, W, device=x.device)

        x_cat = torch.cat([x, past], dim=1)  # shape: [B, input_dim+past_channels, H, W]
        x_stage1 = self.coarse_generator(x_cat, mask)
        x_stage2, offset_flow = self.fine_generator(x_cat, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow

class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None, dropout_p=0.05):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2, dropout_p=dropout_p)
        self.conv2_downsample = gen_conv(cnum, cnum * 2, 3, 2, 1, dropout_p=dropout_p)
        self.conv3 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1, dropout_p=dropout_p)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1, dropout_p=dropout_p)
        self.conv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2, dropout_p=dropout_p)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4, dropout_p=dropout_p)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8, dropout_p=dropout_p)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16, dropout_p=dropout_p)
        self.conv11 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.conv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.conv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1, dropout_p=dropout_p)
        self.conv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1, dropout_p=dropout_p)
        self.conv15 = gen_conv(cnum * 2, cnum, 3, 1, 1, dropout_p=dropout_p)
        self.conv16 = gen_conv(cnum, cnum // 2, 3, 1, 1, dropout_p=dropout_p)
        self.conv17 = gen_conv(cnum // 2, 1, 3, 1, 1, activation='none', dropout_p=0.0)

    def forward(self, x, mask):
        B, _, H, W = x.size()
        ones = torch.ones(B, 1, H, W, device=x.device)
        mask = mask.to(x.device)
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x_stage1 = torch.clamp(x, -1., 1.)
        return x_stage1

class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None, dropout_p=0.05):
        super(FineGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2, dropout_p=dropout_p)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1, dropout_p=dropout_p)
        self.conv3 = gen_conv(cnum, cnum * 2, 3, 1, 1, dropout_p=dropout_p)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 2, 3, 2, 1, dropout_p=dropout_p)
        self.conv5 = gen_conv(cnum * 2, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2, dropout_p=dropout_p)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4, dropout_p=dropout_p)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8, dropout_p=dropout_p)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16, dropout_p=dropout_p)
        # attention branch
        self.pmconv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2, dropout_p=dropout_p)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1, dropout_p=dropout_p)
        self.pmconv3 = gen_conv(cnum, cnum * 2, 3, 1, 1, dropout_p=dropout_p)
        self.pmconv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1, dropout_p=dropout_p)
        self.pmconv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.pmconv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, activation='relu', dropout_p=dropout_p)
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=self.use_cuda, device_ids=self.device_ids)
        self.pmconv9 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.pmconv10 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.allconv11 = gen_conv(cnum * 8, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.allconv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, dropout_p=dropout_p)
        self.allconv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1, dropout_p=dropout_p)
        self.allconv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1, dropout_p=dropout_p)
        self.allconv15 = gen_conv(cnum * 2, cnum, 3, 1, 1, dropout_p=dropout_p)
        self.allconv16 = gen_conv(cnum, cnum // 2, 3, 1, 1, dropout_p=dropout_p)
        self.allconv17 = gen_conv(cnum // 2, 1, 3, 1, 1, activation='none', dropout_p=0.0)

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        B, _, H, W = xin.size()
        ones = torch.ones(B, 1, H, W, device=xin.device)
        mask = mask.to(xin.device)
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)
        return x_stage2, offset_flow


class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        raw_int_fs = list(f.size())   # [B, C, H, W]
        raw_int_bs = list(b.size())
        kernel = 2 * self.rate
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate * self.stride, self.rate * self.stride],
                                      rates=[1, 1],
                                      padding='same')
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)
        raw_w_groups = torch.split(raw_w, 1, dim=0)
        f = F.interpolate(f, scale_factor=1. / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1. / self.rate, mode='nearest')
        int_fs = list(f.size())
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)
        w_groups = torch.split(w, 1, dim=0)
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1. / (4 * self.rate), mode='nearest')
        int_ms = list(mask.size())
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)
        m = m[0]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3)
        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale
        fuse_weight = torch.eye(k).view(1, 1, k, k)
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()
        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])
            yi = F.conv2d(xi, wi_normed, stride=1)
            if self.fuse:
                yi = yi.view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            yi = yi * mm
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm
            offset = torch.argmax(yi, dim=1, keepdim=True)
            if int_bs != int_fs:
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset // int_fs[3], offset % int_fs[3]], dim=1)
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.
            y.append(yi)
            offsets.append(offset)
        y = torch.cat(y, dim=0)
        y = y.contiguous().view(raw_int_fs)
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_bs[0], 2, *int_bs[2:])
        h_add = torch.arange(int_bs[2]).view([1, 1, int_bs[2], 1]).expand(int_bs[0], -1, -1, int_bs[3])
        w_add = torch.arange(int_bs[3]).view([1, 1, 1, int_bs[3]]).expand(int_bs[0], -1, int_bs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        if self.use_cuda:
            ref_coordinate = ref_coordinate.cuda()
        offsets = offsets - ref_coordinate
        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        if self.use_cuda:
            flow = flow.cuda()
        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate * 4, mode='nearest')
        return y, flow


class LocalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum * 4 * 16 * 16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum * 4 * 16 * 16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum * 2, 5, 2, 2)
        self.conv3 = dis_conv(cnum * 2, cnum * 4, 5, 2, 2)
        self.conv4 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu', dropout_p=0.0):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation, dropout_p=dropout_p)

def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu', dropout_p=0.0):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation, dropout_p=dropout_p)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False, dropout_p=0.0):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            raise ValueError(f"Unsupported padding type: {pad_type}")
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization: {norm}")
        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            raise ValueError(f"Unsupported weight normalization: {weight_norm}")
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
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
            raise ValueError(f"Unsupported activation: {activation}")
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)
        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)
        if dropout_p > 0.0:
            self.dropout = nn.Dropout2d(p=dropout_p)
        else:
            self.dropout = None
        self.dropout_p = dropout_p

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)


import torch.nn.utils.prune as prune

def prune_generator(model, amount=0.3):

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # 对权重做结构化剪枝，剪掉 amount 比例的输出通道
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    return model

def remove_prune_reparam(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)