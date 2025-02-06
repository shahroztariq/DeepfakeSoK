#!/usr/bin/env python3

import torch
import torch.nn as nn
from .resnet import resnet34, conv3x3
from .CADDM.efficientnet_pytorch import EfficientNet


class ADM_ExtraBlock(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes,
            kernel_size=3, stride=1, downsample=None
    ):
        super(ADM_ExtraBlock, self).__init__()
        # stride/2 maybe applied on conv1
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=stride)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Conv + BatchNorm + RelU
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample: feature Map size/2 || Channel increase
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ADM_EndBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1):
        super(ADM_EndBlock, self).__init__()
        # stride/2 maybe applied on conv1
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=stride)

        self.relu = nn.ReLU(inplace=True)
        # Conv + BatchNorm + RelU
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)

        self.downsample = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += self.downsample(residual)
        out = self.relu(out)

        return out


class Multi_scale_Detection_Module(nn.Module):

    def __init__(
            self, inplanes, class_num=2,
            width_hight_ratios=2, extra_layers=None
    ):
        super(Multi_scale_Detection_Module, self).__init__()

        # Multi-scale Detection Module.

        multi_scale_detector = list()
        multi_scale_classifier = list()

        for extra_block in extra_layers:
            ks = 3 if extra_block != ADM_EndBlock else 1
            pad = 1 if extra_block != ADM_EndBlock else 0

            multi_scale_classifier.append(
                nn.Conv2d(
                    inplanes, width_hight_ratios*class_num,
                    kernel_size=ks, stride=1, padding=pad
                )
            )

            multi_scale_detector.append(
                nn.Conv2d(
                    inplanes, width_hight_ratios*4,
                    kernel_size=ks, stride=1, padding=pad
                )
            )

        self.ms_dets = nn.ModuleList(multi_scale_detector)
        self.ms_cls = nn.ModuleList(multi_scale_classifier)

    def forward(self, x):
        confidence, location = list(), list()
        for (feat, detector, classifier) in zip(x, self.ms_dets, self.ms_cls):
            location.append(detector(feat).permute(0, 2, 3, 1).contiguous())
            confidence.append(classifier(feat).permute(0, 2, 3, 1).contiguous())

        confidence = torch.cat([o.view(o.size(0), -1) for o in confidence], 1)
        location = torch.cat([o.view(o.size(0), -1) for o in location], 1)

        return location, confidence


class Artifact_Detection_Module(nn.Module):

    def __init__(
            self, inplanes, blocks=1, class_num=2,
            width_hight_ratios=2, extra_layers=None,
    ):

        super(Artifact_Detection_Module, self).__init__()

        # Artifact Detection Module Extra Layers.

        self.cls_num = class_num
        self.inplanes = inplanes

        adm_extra_layers = list()

        if extra_layers is None:
            extra_layers = [ADM_ExtraBlock] * 3 + [ADM_EndBlock]

        for i, extra_block in enumerate(extra_layers):
            ks = 3 if i else 1
            if extra_block != ADM_EndBlock:
                adm_extra_layers.append(
                    self._make_layer(
                        extra_block, inplanes,
                        blocks=blocks, kernel_size=ks, stride=1
                    )
                )
            else:
                adm_extra_layers.append(extra_block(inplanes, inplanes))

        self.adm_extra_layers = nn.ModuleList(adm_extra_layers)

        self.multi_scale_detection_module = Multi_scale_Detection_Module(
            inplanes, extra_layers=extra_layers
        )

    def _make_layer(self, block, planes, blocks, kernel_size, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion)
        )

        layers = []
        layers.append(block(
            self.inplanes, planes * block.expansion, kernel_size=kernel_size,
            stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes * block.expansion, ))

        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        adm_feats = list()

        for adm_layer in self.adm_extra_layers:
            x = adm_layer(x)
            adm_feats.append(x)

        location, confidence = self.multi_scale_detection_module(adm_feats)

        location = location.view(bs, -1, 4)
        confidence = confidence.view(bs, -1, self.cls_num)

        adm_final_feat = adm_feats[-1]

        return location, confidence, adm_final_feat

# vim: ts=4 sw=4 sts=4 expandtab


class CADDM(nn.Module):

    def __init__(self, num_classes, backbone='resnet34'):
        super(CADDM, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone

        if backbone == 'resnet34':
            self.base_model = resnet34(pretrained=True)
            self.inplanes = self.base_model.out_num_features
        elif backbone == 'efficientnet-b3':
            self.base_model = EfficientNet.from_pretrained(
                'efficientnet-b3', out_size=[1, 3]
            ) 
            
            self.inplanes = self.base_model.out_num_features
			# self.base_model = EfficientNet.from_pretrained('efficientnet-b3', advprop=True, num_classes=1000)
            # self.inplanes = self.base_model._fc.in_features
        elif backbone == 'efficientnet-b4':
            self.base_model = EfficientNet.from_pretrained(
                'efficientnet-b4', out_size=[1, 3]
            ) 
            self.inplanes = self.base_model.out_num_features
            # self.base_model =  EfficientNet.from_pretrained('efficientnet-b4', advprop=True, num_classes=1000)
            # self.inplanes = self.base_model._fc.in_features
        else:
            raise ValueError("Unsupported Backbone!")

        

        self.adm = Artifact_Detection_Module(self.inplanes)

        self.fc = nn.Linear(self.inplanes, num_classes)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_num = x.size(0)
        x, global_feat = self.base_model(x)

        # location result, confidence of each anchor, final feature map of adm.
        loc, cof, adm_final_feat = self.adm(x)

        final_cls_feat = global_feat + adm_final_feat
        final_cls = self.fc(final_cls_feat.view(batch_num, -1))

        if self.training:
            return loc, cof, final_cls
        return self.softmax(final_cls)
if __name__ == '__main__':
    model  = CADDM(2, backbone='resnet34')
    pretrained_model = '//media/NAS/USERS/binh/DeepfakeCampaign/pretrained_checkpoints/iil/resnet34.pkl'
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['network'])
    model.eval()
    batch_img = torch.randn(32, 3, 224, 224)
    output = model(batch_img)
    print("Test success: ", output.shape)