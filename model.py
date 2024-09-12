import torch
import torch.nn as nn
import torch.nn.functional as F
from CSwin import cswin_small,cswin_tiny,CSWinBlock
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
import numpy as np


def upsize(x, scale_factor=2):
    # x = F.interpolate(x, size=e.shape[2:], mode='nearest')
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x

class eca_layer(nn.Module):

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x)


        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)


        y = self.sigmoid(y)

        return x * y.expand_as(x)




class decoderbox(nn.Module):#n,c,h,w -> n,c/2,2h,2w
    def __init__(self,in_planes,out_planes):
        super(decoderbox,self).__init__()
        #b,c,h,w -> b,c/4,h,w
        self.eca = eca_layer(channel=in_planes)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(in_planes,in_planes // 4, kernel_size=3,stride=1,padding=1,bias=False)
        self.norm1 = nn.LayerNorm(in_planes // 4,eps=1e-6)
        #n,c,h,w -> n,c/4,2h,2w
        self.deconv = nn.ConvTranspose2d(in_planes // 4, in_planes // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.LayerNorm(in_planes // 4, eps=1e-6)
        #n,c/4,h,w -> n,c/2,h,w
        self.conv2 = nn.Conv2d(in_planes // 4, out_planes, kernel_size=3,stride=1,padding=1,bias=False)
        self.norm3 = nn.LayerNorm(out_planes, eps=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):
        #first conv
        x = self.eca(x)
        x = self.conv1(x)
        x = x.permute(0,2,3,1) #n,c,h,w -> n,h,w,c
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0,3,1,2)#n,h,w,c -> n,c,h,w

        #c,h,w -> c/4,2h,2w

        x = self.deconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        # n,c/4,2h,2w -> n,c/2,2h,2w
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm3(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        return x

class lastconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(lastconv, self).__init__()
        # down
        self.act = nn.GELU()
        self.finalconv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.LayerNorm(out_planes, eps=1e-6)

        self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # linear att
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(out_planes, 4 * out_planes, bias=False)
        self.linear2 = nn.Linear(4 * out_planes, out_planes, bias=False)

        self.att = nn.Sigmoid()

    def forward(self, x):
        x = self.finalconv(x)
        x = x.permute(0, 2, 3, 1)  # n,c,h,w -> n,h,w,c
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # n,h,w,c -> n,c,h,w
        x = self.act(x)
        input = x
        #print(input.shape)
        x = self.conv3(x)
        x = x.permute(0, 2, 3, 1)  # n,c,h,w -> n,h,w,c
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # n,h,w,c -> n,c,h,w
        x = self.act(x)

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # n,c,h,w -> n,h,w,c
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # n,h,w,c -> n,c,h,w
        x = F.softmax(x,dim=-1)
        x = x.transpose(2, 3)
        x = input @ x
        #print(x.shape)
        _,_,h,w = x.size()

        y1 = self.avgpool(input)
        y2 = self.maxpool(input)
        y = y1 + y2
        n, c, _, _ = y.size()
        y = y.view(n, c)
        y = self.linear1(y)
        y = self.act(y)
        y = self.linear2(y)
        y = self.att(y)

        y = y.view(n,c,1,1)
        #print(y.shape)

        mul = x * y

        return mul
class double_conv(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(double_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,out_planes,kernel_size=3,padding=1)
        self.norm = nn.LayerNorm(out_planes,eps=1e-6)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_planes,out_planes,kernel_size=3,padding=1)

    def forward(self,x):

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # n,c,h,w -> n,h,w,c
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # n,h,w,c -> n,c,h,w
        x = self.act(x)

        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)  # n,c,h,w -> n,h,w,c
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # n,h,w,c -> n,c,h,w
        x = self.act(x)

        return x
class outconv(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(outconv, self).__init__()

        self.conv = nn.ConvTranspose2d(in_planes,out_planes,3, stride=2, padding=1, output_padding=1)
        self.norm = nn.LayerNorm(out_planes,eps=1e-6)
        self.act = nn.GELU()

    def forward(self,x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # n,c,h,w -> n,h,w,c
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # n,h,w,c -> n,c,h,w
        x = self.act(x)

        return x


class down(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_planes, out_planes)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x
class MergeBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(MergeBlock, self).__init__()
        self.norm = norm_layer(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x

class Rshape_Norm(nn.Module):
    def __init__(self,curr_dim,H_W_size):
        super().__init__()
        self.curr_dim = curr_dim
        self.H_W_size = H_W_size

        self.norm =nn.LayerNorm(curr_dim)

    def forward(self,x):
        x = self.norm(x)
        B, L, C = x.shape
        H = W = self.H_W_size
        assert L == H * W, "L must be equal to H*W, where H and W have the same value."


        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x


class csunet(nn.Module):
    def __init__(self,num_channels,num_class,num_cls,num_heads=[2,2,4,8],depth=[1,1,1,1],mlp_ratio=4.,
                 qkv_bias=False,qk_scale=None,attn_drop_rate=0.,split_size=[1,2,7,7],drop_rate=0.
                 ,drop_path_rate=0.1,norm_layer=nn.LayerNorm):
        super(csunet, self).__init__()

        self.backbone = cswin_small()
        heads = num_heads# [64, 128, 256, 512]
        self.num_channels = num_channels
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]
        self.num_class = num_class
        self.num_cls = num_cls
        path = 'cswin_small_224.pth'
        save_model = torch.load(path)
        # print(save_model['state_dict_ema'].keys())
        model_dict = self.backbone.state_dict()
        self.resnet = models.resnet34(pretrained=True)
        self.initial_layers = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # print(model_dict.keys())
        state_dict = {k: v for k, v in save_model['state_dict_ema'].items() if k in model_dict.keys()}
        # print(state_dict)
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.mix = nn.Parameter(torch.FloatTensor(7))
        self.mix.data.fill_(1)
        self.midconv = lastconv(512,512)
        self.multiway_4 = nn.ModuleList(
            [CSWinBlock(dim=512, num_heads=heads[3], patches_resolution=224 // 32, mlp_ratio=mlp_ratio
                        , qkv_bias=True, qk_scale=None, split_size=split_size[-1], drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer)
             for i in range(depth[-1])])
        self.norm4 = nn.LayerNorm(512)
        self.merge_4 = MergeBlock(dim=512)
        self.re_norm_4 = Rshape_Norm(curr_dim=512, H_W_size=7)
        self.multiway_3 = nn.ModuleList(
            [CSWinBlock(dim=256, num_heads=heads[2], patches_resolution=224 // 16, mlp_ratio=mlp_ratio
                        , qkv_bias=True, qk_scale=None, split_size=split_size[2], drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
             for i in range(depth[2])])
        self.merge_3 = MergeBlock(dim=256)
        self.norm3 = nn.LayerNorm(256)
        self.re_norm_3 = Rshape_Norm(curr_dim=256, H_W_size=14)
        self.multiway_2 = nn.ModuleList(
            [CSWinBlock(dim=128, num_heads=heads[1], patches_resolution=224 // 8, mlp_ratio=mlp_ratio
                        , qkv_bias=True, qk_scale=None, split_size=split_size[1], drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
             for i in range(depth[1])])
        self.merge_2 = MergeBlock(dim=128)
        self.norm2 = nn.LayerNorm(128)
        self.re_norm_2 = Rshape_Norm(curr_dim=128, H_W_size=28)

        self.multiway_1 = nn.ModuleList(
            [CSWinBlock(dim=64, num_heads=heads[0], patches_resolution=224 // 4, mlp_ratio=mlp_ratio
                        , qkv_bias=True, qk_scale=None, split_size=split_size[0], drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
             for i in range(depth[0])])
        self.merge_1 = MergeBlock(dim=64)
        self.norm1 = nn.LayerNorm(64)
        self.re_norm_1 = Rshape_Norm(curr_dim=64, H_W_size=56)


        self.up5 = decoderbox(512,256)
        self.up4 = decoderbox(256,128)
        self.up3 = decoderbox(128,64)
        self.up2 = decoderbox(64,1)
        self.upconv5 = double_conv(256,256)
        self.upconv4 = double_conv(128,128)
        self.upconv3 = double_conv(64,64)
        self.upconv2 = double_conv(1,1)
        self.outconv = outconv(1,num_class)
        self.logit1 = nn.Conv2d(1,num_class,kernel_size=1)
        self.logit2 = nn.Conv2d(64,num_class,kernel_size=1)
        self.logit3 = nn.Conv2d(128,num_class,kernel_size=1)
        self.logit0 = nn.Conv2d(1,num_class,kernel_size=1)
        self.logit5 = nn.Conv2d(512,num_class,kernel_size=1)
        self.logit6 = nn.Conv2d(256,num_class,kernel_size=1)






    def forward(self,x,superpixel):
        _, _, H, W = x.shape

        x0 = self.initial_layers(superpixel)
        x1 = self.layer1(x0)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        cswin = self.backbone(x)

        e1 = cswin[0]#64 48 48
        e1  = 0.8 * e1 + 0.2*x1
        e1 = self.merge_1(e1)
        for block1 in self.multiway_1:
            block1.H = 56
            block1.W = 56
            cs_out1 = block1(e1)
        e1 = self.norm1(cs_out1)
        e1 = self.re_norm_1(e1)


        e2 = cswin[1]#128 24 24
        e2 = 0.8 *e2 + 0.2 *x2
        e2 = self.merge_2(e2)
        for block2 in self.multiway_2:
            block2.H = 28
            block2.W = 28
            cs_out2 = block2(e2)
        e2 = self.norm2(cs_out2)
        e2 = self.re_norm_2(e2)

        e3 = cswin[2]#256 12 12
        e3 = 0.8 *e3 + 0.2 *x3
        e3 = self.merge_3(e3)
        for block3 in self.multiway_3:
            block3.H = 14
            block3.W = 14
            cs_out3 = block3(e3)
        e3 = self.norm3(cs_out3)
        e3 = self.re_norm_3(e3)

        e4 = cswin[3]#512 6  6
        e4 = 0.8 *e4 + 0.2 *x4
        e4 = self.merge_4(e4)
        for block4 in self.multiway_4:
            block4.H = 7
            block4.W = 7
            cs_out4 = block4(e4)
        e4 = self.norm4(cs_out4)
        e4 = self.re_norm_4(e4)



        e5 = self.midconv(e4)#512,3,3
        #up
        up5 = self.up5(e5) #256
        up5 = up5 + e3
        up5 = self.upconv5(up5)#256
        up4 = self.up4(up5)#128
        up4 = up4 + e2
        up4 = self.upconv4(up4)
        up3 = self.up3(up4)#128,24,24
        up3 = up3+e1#256,24,24
        up3 = self.upconv3(up3)#128,24,24
        up2 = self.up2(up3)#64,48,48
        up2 = self.upconv2(up2)
        out = self.outconv(up2)#1,96,96
        logit1 = self.logit1(up2)
        logit1 = F.interpolate(logit1, size=(H, W), mode='bilinear', align_corners=False)
        logit2 = self.logit2(up3)
        logit2 = F.interpolate(logit2, size=(H, W), mode='bilinear', align_corners=False)
        logit3 = self.logit3(up4)
        logit3 = F.interpolate(logit3, size=(H, W), mode='bilinear', align_corners=False)
        logit0 = self.logit0(out)
        logit0 = F.interpolate(logit0, size=(H, W), mode='bilinear', align_corners=False)
        logit5 = self.logit5(e5)
        logit5 = F.interpolate(logit5, size=(H, W), mode='bilinear', align_corners=False)
        logit6 = self.logit6(up5)
        logit6 = F.interpolate(logit6, size=(H, W), mode='bilinear', align_corners=False)


        logit = self.mix[1] * logit1 + self.mix[2] * logit2 + self.mix[3] * logit3 + self.mix[4] * logit0 + self.mix[5] * logit5 + self.mix[6] * logit6

        return  logit

if __name__ == "__main__":
    from thop import profile
    input = torch.randn(8, 3, 224, 224)
    superpixel = torch.randn(8, 3, 224, 224)
    model = csunet(num_channels=3,num_class=1,num_cls=3)



    for name,param in model.named_parameters():
        if param.requires_grad:
            print(name)
    output = model(input,superpixel)# 8*64*1*1
