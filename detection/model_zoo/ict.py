import os
from torch import nn
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()


        # with open('runs/%s/config.pkl'%name,'rb') as f:
        #     config = pickle.load(f)
        self.net =  combface_base_patch8_112().cpu()
        model_path = "//media/NAS/USERS/binh/DeepfakeCampaign/pretrained_checkpoints/ict/"
        state_dict = torch.load(model_path + 'ICT_Base.pth', map_location='cpu')

        
        self.net.load_state_dict(state_dict['model'], strict=True) # This one make sure you load a correct model

        print("Load reference ...")
        drop = 1
        self.query = {}
        files = os.listdir(model_path)
        temp_inner = []
        temp_outer = []
        for fi in files:
            if 'ref.pkl' == fi:
                temp_query = torch.load(os.path.join(model_path, fi))
                print ('Loading:', fi, 'Drop:', 1-drop)
                bz = temp_query['inner'].shape[0]
                idx_shuffle = torch.randperm(bz)[:int(bz * drop)]
                temp_inner.append(temp_query['inner'][idx_shuffle])
                temp_outer.append(temp_query['outer'][idx_shuffle])
        if len(temp_inner) == 0:
            print ('No reference set found, ICT-Ref can not work.')
            exit(0)
        self.query['inner'] = torch.cat(temp_inner, 0).cuda().unsqueeze(0)
        
        self.query['outer'] = torch.cat(temp_outer, 0).cuda().unsqueeze(0)

        print("Shape of query['inner']: ", self.query['inner'].size())
        print("Shape of query['outer']: ", self.query['outer'].size())

        
    def forward(self,x, return_ft=False):
        x = self.net.forward(x, return_ft)
        return x


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class CombFaceVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outer_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head = None

        self.bn = nn.BatchNorm1d(self.embed_dim)

        trunc_normal_(self.outer_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        outer_token = self.outer_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, outer_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x, return_ft=False):
        inner, outer = self.forward_features(x)
        inner = self.bn(inner.float())
        outer = self.bn(outer.float())
        return l2_norm(inner), l2_norm(outer)


def combface_base_patch8_112(pretrained=False, **kwargs):
    model = CombFaceVisionTransformer(
        img_size=112, patch_size=8, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    model = Detector().cuda()
    print("Successfully loaded ")
    x = torch.randn(8, 3, 112, 112).cuda()
    y_pred = model(x)
    for y in y_pred:
        print(y.shape)