#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

""" 
    MTI-Net implementation based on HRNet backbone 
    https://arxiv.org/pdf/2001.06902.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.seg_hrnet import hrnet_w18, hrnet_w48, HighResolutionHead, HighResolutionHeadwoCat
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone, backbone_channels, tasks: list, class_nb=13):
        super(MultiTaskModel, self).__init__()
        # assert(set(decoders.keys()) == set(tasks))
        # Backbone
        self.backbone = backbone
        # backbone_channels = [48, 96, 192, 384]
        self.tasks = tasks
        # Feature aggregation through HRNet heads
        self.semseg_head = HighResolutionHead(backbone_channels, num_outputs=class_nb)
        self.depth_head = HighResolutionHead(backbone_channels, num_outputs=1)
        self.normal_head = HighResolutionHead(backbone_channels, num_outputs=3)

        self.class_nb = class_nb

    def forward(self, x, masks=None):
        out = {}
        out_size = x.size()[2:]

        # if masks is not None:
        #     x_masks = torch.cat((x, masks), dim=1)
        #     x = self.refinenet(x_masks)

        x = self.backbone(x)
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        feat = torch.cat([x[0], x1, x2, x3], 1)

        for i, t in enumerate(self.tasks):
            if t == 'semantic':
                out[t] = F.interpolate(self.semseg_head(x), out_size, mode = 'bilinear')
            elif t == 'depth':
                out[t] = F.interpolate(self.depth_head(x), out_size, mode = 'bilinear')
            elif t == 'normal':
                out[t] = F.interpolate(self.normal_head(x), out_size, mode = 'bilinear')
            else:
                RuntimeError('Wrong Head')

        return out, feat

    
def get_MTL(tasks, class_nb=13):
    backbone_channels = [18, 36, 72, 144]
    backbone = hrnet_w18(pretrained=True)

    return MultiTaskModel(backbone, backbone_channels, tasks, class_nb=class_nb)
