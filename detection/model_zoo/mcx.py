import torch
from torch import nn
import torchvision
from torchvision import models
import numpy as np
import sys
import torch.nn.functional as F
# from utils import init_weights_zero, init_weights_xavier_uniform, init_weights_xavier_normal, init_weights_kaiming_uniform, init_weights_kaiming_normal
# from model.vit import vit_b_16
# from resnet_multichannel import get_arch as Resnet_multi
# from xception import xception

# from xception_multichannel import xception_multichannels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# __all__ = ['xception']
#
# model_urls = {
#     'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
# }


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, num_classes)

        # ------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def xception(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    if pretrained:
        checkpoint = torch.load('xception-c0a72b38.pth.tar')
        model.load_state_dict(checkpoint)
        # model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model

def init_weights_zero(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.zeros_(m.weight)


def init_weights_xavier_normal(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


def init_weights_xavier_uniform(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def init_weights_kaiming_normal(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)


def init_weights_kaiming_uniform(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        
def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def pw_cosine_distance(vector):
    normalized_vec = F.normalize(vector)
    res = torch.mm(normalized_vec, normalized_vec.T)
    cos_dist = 1 - res
    return cos_dist


class API_Net(nn.Module):
    def __init__(self, num_classes=5, model_name='res101', weight_init='pretrained'):
        super(API_Net, self).__init__()

        # ---------Resnet101---------
        if model_name == 'res101':
            model = models.resnet101(pretrained=True)
            kernel_size = 14
        # layers = list(resnet101.children())[:-2]
        elif model_name == 'res101_9ch':
            raise ValueError('res101_9ch is not supported')
            resnet101_9_channel = Resnet_multi(101, 9)
        # use resnet34_4_channels(False) to get a non pretrained model
            model = resnet101_9_channel(True)
            kernel_size = 14
        elif model_name == 'res101_6ch':
            raise ValueError('res101_6ch is not supported')
            
            resnet101_6_channel = Resnet_multi(101, 6)
            model = resnet101_6_channel(True)
            kernel_size = 14

        # ---------Efficientnet---------
        elif model_name == 'effb0':
            model = models.efficientnet_b0(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb1':
            model = models.efficientnet_b1(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb2':
            model = models.efficientnet_b2(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb3':
            model = models.efficientnet_b3(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb4':
            model = models.efficientnet_b4(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb5':
            model = models.efficientnet_b5(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb6':
            model = models.efficientnet_b6(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb7':
            model = models.efficientnet_b7(pretrained=True)
            kernel_size = 14

        # ---------Xception---------
        elif model_name == 'xception':
            model = xception()
            kernel_size = 14
        elif model_name == 'xception_9channels':
            raise ValueError('xception_9channels is not supported')
            model = xception_multichannels()
            kernel_size = 14
        elif model_name == 'xception_6channels':
            raise ValueError('xception_6channels is not supported')
            
            model = xception_multichannels(channel_num=6)
            kernel_size = 14

        # ---------Vision Transformer---------
        # elif model_name == 'vit_b_16':
        #     model = vit_b_16(pretrained=True)
        #     kernel_size = 28

        else:
            sys.exit('wrong model name baby')

        if weight_init == 'zero':
            model.apply(init_weights_zero)
            print('init weight 0')
        elif weight_init == 'xavier_uniform':
            print('init weight xavier uniform')
            model.apply(init_weights_xavier_uniform)
        elif weight_init == 'xavier_normal':
            print('init weight xavier normal')
            model.apply(init_weights_xavier_normal)
        elif weight_init == 'kaiming_uniform':
            print('init weight kaiming uniform')
            model.apply(init_weights_kaiming_uniform)
        elif weight_init == 'kaiming_normal':
            print('init weight kaiming normal')
            model.apply(init_weights_kaiming_normal)

        else:
            print('you are using pretrained model if you do not load the parameter')


        layers = list(model.children())[:-2]
        if 'res' in model_name:
            fc_size = model.fc.in_features
        elif 'eff' in model_name:
            fc_size = model.classifier[1].in_features
        elif 'vit' in model_name:
            fc_size = model.hidden_dim
        elif 'xception' in model_name:
            fc_size = 2048
        else:
            sys.exit('wrong network name baby')

        self.conv = nn.Sequential(*layers)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1)

        self.map1 = nn.Linear(fc_size * 2, 512)
        self.map2 = nn.Linear(512, fc_size)
        self.fc = nn.Linear(fc_size, num_classes)

        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        # wrong 9-channel
        # self.conv_reduce = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1)
        # --- to here

    def forward(self, images, targets=None, flag='test', dist_type='euclidean', loader='three'):
        # wrong 9-channel ---
        if loader == 'nine_channels':
            images = self.conv_reduce(images)
        # --- to here
        # print(f'images {images.shape}')
        conv_out = self.conv(images)
        pool_out_old = self.avg(conv_out)
        pool_out = pool_out_old.squeeze()

        if flag == 'train':
            intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(pool_out, targets, dist_type)

            features1 = torch.cat([pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat([pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)
            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)

            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1

            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2

            logit1_self = self.fc(self.drop(features1_self))
            logit1_other = self.fc(self.drop(features1_other))
            logit2_self = self.fc(self.drop(features2_self))
            logit2_other = self.fc(self.drop(features2_other))

            features = self.fc(pool_out)

            return logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2, features

        elif flag == 'features':
            intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(pool_out, targets, dist_type)

            features1 = torch.cat([pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat([pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)
            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)

            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1

            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2

            return features1_self, features1_other, features2_self, features2_other, labels1, labels2

        elif flag == 'val':
            return self.fc(pool_out)
        elif flag == 'test':
            return self.fc(pool_out)
        elif flag == 'tsne':
            return pool_out


    def get_pairs(self, embeddings, labels, dist_type):
        # print(f'embedding shape {embeddings.shape}')
        if dist_type == 'euclidean':
            distance_matrix = pdist(embeddings).detach().cpu().numpy()
        elif dist_type == 'cosine':
            distance_matrix = pw_cosine_distance(embeddings).detach().cpu().numpy()
        else:
            sys.exit('wrong distance name baby')

        labels = labels.detach().cpu().numpy().reshape(-1,1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = (labels == labels.T)
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs == False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])
        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]

            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]

        intra_labels = torch.from_numpy(intra_labels).long().to(device)
        intra_pairs = torch.from_numpy(intra_pairs).long().to(device)
        inter_labels = torch.from_numpy(inter_labels).long().to(device)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(device)

        return intra_pairs, inter_pairs, intra_labels, inter_labels



class API_Net_gradcam(nn.Module):
    def __init__(self, num_classes=5, model_name='res101', weight_init='pretrained'):
        super(API_Net_gradcam, self).__init__()

        # ---------Resnet101---------
        if model_name == 'res101':
            model = models.resnet101(pretrained=True)
            kernel_size = 14
        # layers = list(resnet101.children())[:-2]
        elif model_name == 'res101_9ch':
            raise ValueError('res101_9ch is not supported')
            
            resnet101_9_channel = Resnet_multi(101, 9)
        # use resnet34_4_channels(False) to get a non pretrained model
            model = resnet101_9_channel(True)
            kernel_size = 14
        elif model_name == 'res101_6ch':
            raise ValueError('res101_6ch is not supported')
            resnet101_6_channel = Resnet_multi(101, 6)
            model = resnet101_6_channel(True)
            kernel_size = 14

        # ---------Efficientnet---------
        elif model_name == 'effb0':
            model = models.efficientnet_b0(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb1':
            model = models.efficientnet_b1(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb2':
            model = models.efficientnet_b2(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb3':
            model = models.efficientnet_b3(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb4':
            model = models.efficientnet_b4(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb5':
            model = models.efficientnet_b5(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb6':
            model = models.efficientnet_b6(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb7':
            model = models.efficientnet_b7(pretrained=True)
            kernel_size = 14

        # ---------Xception---------
        elif model_name == 'xception':
            model = xception()
            kernel_size = 14
        elif model_name == 'xception_9channels':
            raise ValueError('xception_9channels is not supported')
            model = xception_multichannels()
            kernel_size = 14
        elif model_name == 'xception_6channels':
            raise ValueError('xception_6channels is not supported')
            model = xception_multichannels(channel_num=6)
            kernel_size = 14

        # ---------Vision Transformer---------
        # elif model_name == 'vit_b_16':
        #     model = vit_b_16(pretrained=True)
        #     kernel_size = 28

        else:
            sys.exit('wrong model name baby')

        if weight_init == 'zero':
            model.apply(init_weights_zero)
            print('init weight 0')
        elif weight_init == 'xavier_uniform':
            print('init weight xavier uniform')
            model.apply(init_weights_xavier_uniform)
        elif weight_init == 'xavier_normal':
            print('init weight xavier normal')
            model.apply(init_weights_xavier_normal)
        elif weight_init == 'kaiming_uniform':
            print('init weight kaiming uniform')
            model.apply(init_weights_kaiming_uniform)
        elif weight_init == 'kaiming_normal':
            print('init weight kaiming normal')
            model.apply(init_weights_kaiming_normal)

        else:
            print('you are using pretrained model if you do not load the parameter')


        layers = list(model.children())[:-2]
        if 'res' in model_name:
            fc_size = model.fc.in_features
        elif 'eff' in model_name:
            fc_size = model.classifier[1].in_features
        elif 'vit' in model_name:
            fc_size = model.hidden_dim
        elif 'xception' in model_name:
            fc_size = 2048
        else:
            sys.exit('wrong network name baby')

        self.conv = nn.Sequential(*layers)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1)

        self.map1 = nn.Linear(fc_size * 2, 512)
        self.map2 = nn.Linear(512, fc_size)
        self.fc = nn.Linear(fc_size, num_classes)

        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        # wrong 9-channel
        # self.conv_reduce = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1)
        # --- to here

    def forward(self, images, targets=None):
        conv_out = self.conv(images)
        print(f'conv_out {conv_out.shape}')
        pool_out_old = self.avg(conv_out)
        print(f'pool_out_old {pool_out_old.shape}')
        pool_out = pool_out_old.squeeze()
        print(f'pool_out {pool_out.shape}')

        return conv_out












