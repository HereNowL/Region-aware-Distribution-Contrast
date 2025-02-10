from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from termcolor import colored
#from .sync_bn.inplace_abn.bn import InPlaceABNSync

#BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
BatchNorm2d = functools.partial(nn.BatchNorm2d)
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config['MODEL']['EXTRA']
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = np.sum(pre_stage_channels).astype(int)
    
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        return x

    def init_weights(self, pretrained='',):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            print('Using pretrained weights from location {}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            #for k, _ in pretrained_dict.items():
            #    print('=> loading {} from pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def get_last_layer(self):
        return self.stage4


class HighResolutionFuse(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionFuse, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False))
    
    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x        

class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        
        # 声明三个卷积层，用于计算查询、键和值
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 声明一个softmax层，用于计算注意力矩阵
        self.softmax = nn.Softmax(dim=-1)
        
        # 声明一个卷积层，用于将融合后的特征图进行调整
        self.gamma = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        # 计算查询、键和值
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # 将查询、键进行reshape，以便计算注意力矩阵
        batch_size, channels, height, width = x.size()
        query = query.view(batch_size, -1, height*width).permute(0,2,1)
        key = key.view(batch_size, -1, height*width)
        
        # 计算注意力矩阵
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        
        # 将注意力矩阵和值进行矩阵乘法，得到融合后的特征图
        value = value.view(batch_size, -1, height*width)
        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(batch_size, channels, height, width)
        
        # 将融合后的特征图进行调整，并加权融合原始特征图
        out = self.gamma(out)
        out = x + out
        
        return out
    
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels, 1, 1)
        y = F.relu(self.fc1(y), inplace=True)
        y = self.fc2(y).sigmoid()
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out

class PyramidFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PyramidFusion, self).__init__()
        self.conv_list = nn.ModuleList()
        self.conv_out_list = nn.ModuleList()
        for in_channels in in_channels_list:
            self.conv_list.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            ))
        
    def forward(self, x_list):
        n = len(x_list)
        x = self.conv_list[0](x_list[0])
        out_list = [x]
        for i in range(1, n):
            x = self.conv_list[i](x_list[i])
            y = out_list[-1]
            if x.size()[2:] != y.size()[2:]:
                x = F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True)
            out = x + y
            out_list.append(self.conv_list[i-1](x+y))
        return out_list[-1]

class MSNet(nn.Module):
    def __init__(self, in_channels_list, out_channels, reduction_ratio=16, kernel_size=7):
        super(MSNet, self).__init__()
        self.pyramid_fusion = PyramidFusion(in_channels_list, out_channels)
        self.cbam = CBAM(out_channels, reduction_ratio, kernel_size)

    def forward(self, x_list):
        out = self.pyramid_fusion(x_list)
        out = self.cbam(out)
        return out
    
class HighResolutionHead(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHead, self).__init__()
        last_inp_channels = sum(backbone_channels)
       
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
        
    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x 

class HighResolutionHeadwoCat(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHeadwoCat, self).__init__()
        last_inp_channels = sum(backbone_channels)
       
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
        
    def forward(self, x):
        x = self.last_layer(x)
        return x    
    
class HighResolutionHeadWithCBAM(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHeadWithCBAM, self).__init__()
        out_channels = 48
        # last_inp_channels = sum(backbone_channels)
        last_inp_channels = out_channels * 4
        
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
        # self.cbam = CBAM(last_inp_channels, reduction_ratio=16, kernel_size=7)

        self.cbam_0 = CBAM(out_channels, reduction_ratio=16, kernel_size=7)
        self.cbam_1 = CBAM(out_channels, reduction_ratio=16, kernel_size=7)
        self.cbam_2 = CBAM(out_channels, reduction_ratio=16, kernel_size=7)
        self.cbam_3 = CBAM(out_channels, reduction_ratio=16, kernel_size=7)

        # 声明4个卷积层，用于将输入特征图调整为相同的通道数
        self.conv0 = nn.Conv2d(backbone_channels[0], out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(backbone_channels[1], out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(backbone_channels[2], out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(backbone_channels[3], out_channels, kernel_size=1)
        
    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x_0 = self.conv0(x[0])
        x_1 = self.conv1(x1)
        x_2 = self.conv2(x2)
        x_3 = self.conv3(x3)

        x_0 = self.cbam_0(x_0)
        x_1 = self.cbam_1(x_1)
        x_2 = self.cbam_0(x_2)
        x_3 = self.cbam_1(x_3)

        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)
        x = self.last_layer(x)
        return x    
    
class HighResolutionHeadWithPooling(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHeadWithPooling, self).__init__()
        last_inp_channels = sum(backbone_channels)
       
        self.last_layer = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=last_inp_channels,
            #     out_channels=last_inp_channels,
            #     kernel_size=1,
            #     stride=1,
            #     padding=0),
            # nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
        
        self.pooling = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(self.pooling(x))
        return x  
    
class HighResolutionHeadWithSEModule(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHeadWithSEModule, self).__init__()
        out_channels = 48
        last_inp_channels = out_channels * 4
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
        # 声明4个卷积层，用于将输入特征图调整为相同的通道数
        self.conv0 = nn.Conv2d(backbone_channels[0], out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(backbone_channels[1], out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(backbone_channels[2], out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(backbone_channels[3], out_channels, kernel_size=1)

        self.se0 = SEModule(out_channels)
        self.se1 = SEModule(out_channels)
        self.se2 = SEModule(out_channels)
        self.se3 = SEModule(out_channels)
        
    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x_0 = self.conv0(x[0])
        x_1 = self.conv1(x1)
        x_2 = self.conv2(x2)
        x_3 = self.conv3(x3)

        x_0 = self.se0(x_0)
        x_1 = self.se1(x_1)
        x_2 = self.se2(x_2)
        x_3 = self.se3(x_3)

        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)
        x = self.last_layer(x)
        return x   
    
class HighResolutionHeadWithCrissCrossAttention(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHeadWithCrissCrossAttention, self).__init__()
        # last_inp_channels = sum(backbone_channels)
        out_channels = 8
        last_inp_channels = out_channels * 4
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum = 0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels= num_outputs,
                kernel_size= 1,
                stride = 1,
                padding = 0))
        
        # 声明4个卷积层，用于将输入特征图调整为相同的通道数
        self.conv0 = nn.Conv2d(backbone_channels[0], out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(backbone_channels[1], out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(backbone_channels[2], out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(backbone_channels[3], out_channels, kernel_size=1)

        # 声明4个CrissCrossAttention层，用于融合不同尺度的特征图
        self.cca0 = CrissCrossAttention(out_channels)
        self.cca1 = CrissCrossAttention(out_channels)
        self.cca2 = CrissCrossAttention(out_channels)
        self.cca3 = CrissCrossAttention(out_channels)

        # 声明一个卷积层，用于将融合后的特征图进行调整
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x_0 = self.conv0(x[0])
        x_1 = self.conv1(x[1])
        x_2 = self.conv2(x[2])
        x_3 = self.conv3(x[3])

        # 利用CrissCrossAttention融合不同尺度的特征图
        x_0 = self.cca0(x_0)
        x_1 = self.cca1(x_1)
        x_2 = self.cca2(x_2)
        x_3 = self.cca1(x_3)

        _, _, x0_h, x0_w = x_0.shape
        x_1 = F.interpolate(x_1, (x0_h, x0_w), mode='bilinear')
        x_2 = F.interpolate(x_2, (x0_h, x0_w), mode='bilinear')
        x_3 = F.interpolate(x_3, (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)
        x = self.last_layer(x)
        return x        

def hrnet_w18(pretrained=False):
    import yaml
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]
    hrnet_cfg = os.path.join(PROJECT_ROOT_DIR, 'model', 'model_info', 'hrnet_w18.yml')
     
    with open(hrnet_cfg, 'r') as stream:
        hrnet_cfg = yaml.safe_load(stream)
    
    model = HighResolutionNet(hrnet_cfg)
    if pretrained:
        pretrained_weights = os.path.join(PROJECT_ROOT_DIR, 'model', 'pretrained_models', 'hrnet_w18_small_model_v2.pth')
        if os.path.exists(pretrained_weights):
            model.init_weights(pretrained_weights)
        else:
            raise AssertionError('Error: No pretrained weights found for HRNet18. \n Download weights from https://github.com/HRNet/HRNet-Image-Classification and save them to {}'.format(pretrained_weights))

    return model

def hrnet_w32(pretrained=False):
    import yaml
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]
    hrnet_cfg = os.path.join(PROJECT_ROOT_DIR, 'model', 'model_info', 'hrnet_w32.yml')
     
    with open(hrnet_cfg, 'r') as stream:
        hrnet_cfg = yaml.safe_load(stream)
    
    model = HighResolutionNet(hrnet_cfg)
    if pretrained:
        pretrained_weights = os.path.join(PROJECT_ROOT_DIR, 'model', 'pretrained_models', 'hrnetv2_w32_imagenet_pretrained.pth')
        if os.path.exists(pretrained_weights):
            model.init_weights(pretrained_weights)
        else:
            raise AssertionError('Error: No pretrained weights found for HRNet32. \n Download weights from https://github.com/HRNet/HRNet-Image-Classification and save them to {}'.format(pretrained_weights))

    return model

def hrnet_w48(pretrained=False):
    import yaml
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]
    hrnet_cfg = os.path.join(PROJECT_ROOT_DIR, 'model', 'model_info', 'hrnet_w48.yml')

    with open(hrnet_cfg, 'r') as stream:
        hrnet_cfg = yaml.safe_load(stream)
    
    model = HighResolutionNet(hrnet_cfg)
    if pretrained:
        pretrained_weights = os.path.join(PROJECT_ROOT_DIR, 'model', 'pretrained_models', 'hrnetv2_w48_imagenet_pretrained.pth')
        if os.path.exists(pretrained_weights):
            model.init_weights(pretrained_weights)
        else:
            raise AssertionError('Error: No pretrained weights found for HRNet18. \n Download weights from https://github.com/HRNet/HRNet-Image-Classification and save them to {}'.format(pretrained_weights))

    return model
