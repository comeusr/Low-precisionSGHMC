'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .quantizer import BlockQuantizer

__all__ = ["ResNet18LP", "ResNet34", "ResNet50LP", "ResNet101", "ResNet152"]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, quant, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.quant = quant()
    def forward(self, x):
        out = self.conv1(x)
        out = self.quant(out)
        out = F.relu(self.bn1(out))
        out = self.quant(out)
        out = self.conv2(out)
        out = self.quant(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, quant, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.quant = quant()
    def forward(self, x):
        out = self.conv1(x)
        out = self.quant(out)
        out = F.relu(self.bn1(out))
        out = self.quant(out)
        out = self.conv2(out)
        out = self.quant(out)
        out = F.relu(self.bn2(out))
        out = self.quant(out)
        out = self.conv3(out)
        out = self.quant(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, quant, num_classes, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], quant, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], quant, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], quant, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], quant, stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)
        self.quant = quant()
    def _make_layer(self, block, planes, num_blocks, quant, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, quant, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.quant(out)
        out = F.relu(self.bn1(out))
        out = self.quant(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class ResNet18LP:
    base = ResNet
    args = list()
    kwargs = {"block":BasicBlock, "num_blocks":[2,2,2,2]}

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

class ResNet50LP:
    base = ResNet
    args = list()
    kwargs = {"block":Bottleneck, "num_blocks":[3,4,6,3]}

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

