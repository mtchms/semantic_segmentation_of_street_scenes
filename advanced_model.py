import torch
import torch.nn as nn
import torch.nn.functional as F
# 1

def get_num_groups(channels, max_groups):
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1
    

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=32):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(get_num_groups(F_int, num_groups), F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(get_num_groups(F_int, num_groups), F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None, num_groups=32):
        super(Bottleneck, self).__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(get_num_groups(mid_channels, num_groups), mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(get_num_groups(mid_channels, num_groups), mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(get_num_groups(out_channels, num_groups), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.gn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, layers, num_groups=32):

        super(ResNetEncoder, self).__init__()
        self.num_groups = num_groups
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(get_num_groups(64, num_groups), 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

    def _make_layer(self, mid_channels, blocks, stride):
        downsample = None
        out_channels = mid_channels * Bottleneck.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(get_num_groups(out_channels, self.num_groups), out_channels),
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, mid_channels, stride, downsample, num_groups=self.num_groups))
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, mid_channels, stride=1, downsample=None, num_groups=self.num_groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        
        f1 = self.layer1(x)        
        f2 = self.layer2(f1)       
        f3 = self.layer3(f2)       
        f4 = self.layer4(f3)       
        return f1, f2, f3, f4


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_groups=32):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels, num_groups), out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.GroupNorm(get_num_groups(out_channels, num_groups), out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.GroupNorm(get_num_groups(out_channels, num_groups), out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.GroupNorm(get_num_groups(out_channels, num_groups), out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels, num_groups), out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*5, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels, num_groups), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        out = torch.cat([b1, b2, b3, b4, gp], dim=1)
        out = self.project(out)
        return out  


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=124, layers=[3,4,6,3], num_groups=32, use_attention=True):
        super(DeepLabV3Plus, self).__init__()
        self.use_attention = use_attention
        self.encoder = ResNetEncoder(layers, num_groups=num_groups)
        aspp_in = 512 * Bottleneck.expansion
        self.aspp = ASPP(in_channels=aspp_in, out_channels=256, num_groups=num_groups)
        skip_in = 64 * Bottleneck.expansion
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_in, 48, kernel_size=1, bias=False),
            nn.GroupNorm(get_num_groups(48, num_groups), 48),
            nn.ReLU(inplace=True)
        )
        if use_attention:
            self.att_gate = AttentionGate(F_g=256, F_l=skip_in, F_int=128, num_groups=num_groups)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(256, num_groups), 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(256, num_groups), 256),
            nn.ReLU(inplace=True)
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(256, num_groups), 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        f1, f2, f3, f4 = self.encoder(x)
        aspp_feat = self.aspp(f4)
        aspp_up = F.interpolate(aspp_feat, size=(H//4, W//4), mode='bilinear', align_corners=False)
        skip = f1 
        if self.use_attention:
            skip = self.att_gate(g=aspp_up, x=skip)
        skip = self.skip_conv(skip)  
        x = torch.cat([aspp_up, skip], dim=1) 
        x = self.dec_conv1(x)                 
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        out = self.conv_last(x)  
        return out





