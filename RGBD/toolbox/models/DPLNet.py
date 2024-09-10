import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch
from toolbox.models.segformermodels.backbones.mix_transformer_ourprompt_proj import MITB5
from toolbox.models.segformermodels.backbones.mix_transformer_ourprompt_proj import OverlapPatchEmbed
from functools import partial
from toolbox.models.segformermodels.decode_heads.segformer_head import SegFormerHead

class ADA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0_0 = nn.Linear(dim, dim // 4)
        self.conv0_1 = nn.Linear(dim, dim // 4)
        self.conv = nn.Linear(dim // 4, dim)

    def forward(self, p, x):
        p = self.conv0_0(p)
        x = self.conv0_1(x)
        p1 = p + x
        p1 = self.conv(p1)
        return p1


class DPLNet(nn.Module):
    def __init__(self, in_chans=3, img_size=224, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1):
        super(DPLNet, self).__init__()

        self.rgb = MITB5(pretrained=True)
        self.head = SegFormerHead(4)

        self.learnable_prompt = nn.Parameter(torch.randn(1, 30, 32))

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.ada1 = ADA(dim=64)
        self.ada2 = ADA(dim=128)
        self.ada3 = ADA(dim=320)
        self.ada4 = ADA(dim=512)

        self.conv1 = nn.Conv2d(150, 41, kernel_size=1)

    def forward(self, rgb, depth):
        B = rgb.shape[0]
        outs = []
        learnable_prompt = self.learnable_prompt.expand(rgb.shape[0], -1, -1)
        x1, H, W = self.rgb.patch_embed1(rgb)
        d1, Hd, Wd = self.patch_embed1(depth)
        prompted = self.ada1(d1, x1)
        x1 = x1 + prompted
        for i, blk in enumerate(self.rgb.block1):
            x1 = blk(x1, learnable_prompt, H, W)
        x1 = self.rgb.norm1(x1)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        prompted = prompted.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        x2, H, W = self.rgb.patch_embed2(x1)
        d2, Hd, Wd = self.patch_embed2(prompted)
        prompted = self.ada2(d2, x2)
        x2 = x2 + prompted
        for i, blk in enumerate(self.rgb.block2):
            x2 = blk(x2, learnable_prompt, H, W)
        x2 = self.rgb.norm2(x2)
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        prompted = prompted.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x2)

        x3, H, W = self.rgb.patch_embed3(x2)
        d3, Hd, Wd = self.patch_embed3(prompted)
        prompted = self.ada3(d3, x3)
        x3 = x3 + prompted
        for i, blk in enumerate(self.rgb.block3):
            x3 = blk(x3, learnable_prompt, H, W)
        x3 = self.rgb.norm3(x3)
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        prompted = prompted.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x3)

        x4, H, W = self.rgb.patch_embed4(x3)
        d4, Hd, Wd = self.patch_embed4(prompted)
        prompted = self.ada4(d4, x4)
        x4 = x4 + prompted
        for i, blk in enumerate(self.rgb.block4):
            x4 = blk(x4, learnable_prompt, H, W)
        x4 = self.rgb.norm4(x4)
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x4)

        # x = self.rgb.head(outs)
        x = self.head(outs)
        x = torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.conv1(x)

        return x

if __name__ == '__main__':
    from thop import profile
    x = torch.randn(1, 3, 480, 640)
    ir = torch.randn(1, 3, 480, 640)
    edge = torch.randn(1, 480, 640)
    net = DPLNet()

    # print Flops and learnable parameters
    # out = net(x, ir)
    # for i in out:
    #     print(i.shape)
    # flops, params = profile(net, (x, ir))
    # print('Flops: ', flops / 1e9, 'G')
    # print('Params: ', params / 1e6, 'M')

    # here is the number of learnable parameters.
    s = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(s)
    # print(p)

    # print which parameter are learnable
    # for k, v in net.state_dict().items():
    #     print(k, v.shape)

    x = net(x, ir)
    print(x.shape)
