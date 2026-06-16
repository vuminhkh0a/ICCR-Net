import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))))
    
class MT(nn.Module):
    def __init__(self, with_tsne_emb=False):
        super().__init__()
        self.with_tsne_emb = with_tsne_emb

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = ConvBlock(3, 64)
        self.down_conv2 = ConvBlock(64, 128)
        self.down_conv3 = ConvBlock(128, 256)
        self.down_conv4 = ConvBlock(256, 512)
        self.down_conv5 = ConvBlock(512, 1024)

        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(128, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        down1 = self.down_conv1(x)
        max1 = self.maxpool(down1)
        down2 = self.down_conv2(max1)
        max2 = self.maxpool(down2)
        down3 = self.down_conv3(max2)
        max3 = self.maxpool(down3)
        down4 = self.down_conv4(max3)
        max4 = self.maxpool(down4)
        down5 = self.down_conv5(max4)

        up1 = self.up_conv1(torch.cat([down4, self.up_transpose1(down5)], dim=1))
        up2 = self.up_conv2(torch.cat([down3, self.up_transpose2(up1)], dim=1))
        up3 = self.up_conv3(torch.cat([down2, self.up_transpose3(up2)], dim=1))
        up4 = self.up_conv4(torch.cat([down1, self.up_transpose4(up3)], dim=1))

        out = self.final(up4)

        if self.with_tsne_emb:
            return torch.sigmoid(out), down5
        else: 
            return torch.sigmoid(out)


class Unet(nn.Module):
    def __init__(self, with_tsne_emb=False):
        super().__init__()
        self.with_tsne_emb = with_tsne_emb

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = ConvBlock(3, 64)
        self.down_conv2 = ConvBlock(64, 128)
        self.down_conv3 = ConvBlock(128, 256)
        self.down_conv4 = ConvBlock(256, 512)
        self.down_conv5 = ConvBlock(512, 1024)

        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(128, 64)
    
        self.final = nn.Conv2d(64, 1, kernel_size=1)



    def forward(self, x):
        down1 = self.down_conv1(x)
        max1 = self.maxpool(down1)
        down2 = self.down_conv2(max1)
        max2 = self.maxpool(down2)
        down3 = self.down_conv3(max2)
        max3 = self.maxpool(down3)
        down4 = self.down_conv4(max3)
        max4 = self.maxpool(down4)
        down5 = self.down_conv5(max4)

        up1 = self.up_conv1(torch.cat([down4, self.up_transpose1(down5)], dim=1))
        up2 = self.up_conv2(torch.cat([down3, self.up_transpose2(up1)], dim=1))
        up3 = self.up_conv3(torch.cat([down2, self.up_transpose3(up2)], dim=1))
        up4 = self.up_conv4(torch.cat([down1, self.up_transpose4(up3)], dim=1))

        out = self.final(up4)

        if self.with_tsne_emb:
            return torch.sigmoid(out), down5
        else:
            return torch.sigmoid(out)


# class MT_Proposed(nn.Module):
#     def __init__(self, with_tsne_emb=False, with_att_gate=False):
#         super().__init__()
#         self.with_tsne_emb = with_tsne_emb
#         self.with_att_gate = with_att_gate

#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.down_conv1 = ConvBlock(3, 64)
#         self.down_conv2 = ConvBlock(64, 128)
#         self.down_conv3 = ConvBlock(128, 256)
#         self.down_conv4 = ConvBlock(256, 512)
#         self.down_conv5 = ConvBlock(512, 1024)

#         self.proj_conv = ConvBlock(1024, 1024)
#         self.avg = nn.AdaptiveAvgPool2d(output_size=(1))
#         self.linear = nn.Linear(1024, 8)

#         self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.up_conv1 = ConvBlock(1024, 512)
#         self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.up_conv2 = ConvBlock(512, 256)
#         self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.up_conv3 = ConvBlock(256, 128)
#         self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.up_conv4 = ConvBlock(128, 64)
    
#         self.final = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, x):
#         down1 = self.down_conv1(x)
#         max1 = self.maxpool(down1)
#         down2 = self.down_conv2(max1)
#         max2 = self.maxpool(down2)
#         down3 = self.down_conv3(max2)
#         max3 = self.maxpool(down3)
#         down4 = self.down_conv4(max3)
#         max4 = self.maxpool(down4)
#         down5 = self.down_conv5(max4)

#         b = x.shape[0]
#         proj = self.proj_conv(self.maxpool(down5))
#         proj = proj.contiguous().view(b, -1)
#         cls = self.avg(down5).contiguous().view(b, -1)
#         cls = self.linear(cls)

#         up1 = self.up_conv1(torch.cat([down4, self.up_transpose1(down5)], dim=1))
#         up2 = self.up_conv2(torch.cat([down3, self.up_transpose2(up1)], dim=1))
#         up3 = self.up_conv3(torch.cat([down2, self.up_transpose3(up2)], dim=1))
#         up4 = self.up_conv4(torch.cat([down1, self.up_transpose4(up3)], dim=1))

#         out = torch.sigmoid(self.final(up4))

#         if self.with_tsne_emb:
#             return out, proj, cls, down5
#         else: 
#             return out, proj, cls

# VGG16 BN Unet

class VGG16BN_Unet(nn.Module):

    def __init__(self, with_tsne_emb=False, with_vgg16bn=False):

        super().__init__()

        self.with_tsne_emb = with_tsne_emb
        self.with_vgg16bn = with_vgg16bn
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if not self.with_vgg16bn:

            self.down_conv1 = ConvBlock(3, 64)
            self.down_conv2 = ConvBlock(64, 128)
            self.down_conv3 = ConvBlock(128, 256)
            self.down_conv4 = ConvBlock(256, 512)
            self.down_conv5 = ConvBlock(512, 1024)

        else:
            vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            features = vgg.features

            self.down_conv1 = features[0:6]   
            self.down_conv2 = features[7:13]  
            self.down_conv3 = features[14:23] 
            self.down_conv4 = features[24:33] 
            
            self.down_conv5 = nn.Sequential(features[34:43], ConvBlock(512, 1024))

        
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_conv4 = ConvBlock(128, 64)

        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        down1 = self.down_conv1(x)
        max1 = self.maxpool(down1)

        down2 = self.down_conv2(max1)
        max2 = self.maxpool(down2)

        down3 = self.down_conv3(max2)
        max3 = self.maxpool(down3)

        down4 = self.down_conv4(max3)
        max4 = self.maxpool(down4)

        down5 = self.down_conv5(max4)


        up_t1 = self.up_transpose1(down5)

        up1 = self.up_conv1(torch.cat([down4, up_t1], dim=1))
        up_t2 = self.up_transpose2(up1)
        up2 = self.up_conv2(torch.cat([down3, up_t2], dim=1))
        up_t3 = self.up_transpose3(up2)
        up3 = self.up_conv3(torch.cat([down2, up_t3], dim=1))
        up_t4 = self.up_transpose4(up3)
        up4 = self.up_conv4(torch.cat([down1, up_t4], dim=1))

        out = torch.sigmoid(self.final(up4))
        return out

# ==========================================================
# ICCR-Net
# ==========================================================

class ICCR_NET(nn.Module):

    def __init__(self, with_tsne_emb=False, with_vgg16bn=False, num_class=8):

        super().__init__()

        self.with_tsne_emb = with_tsne_emb
        self.with_vgg16bn = with_vgg16bn
        self.num_class = num_class
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if not self.with_vgg16bn:

            self.down_conv1 = ConvBlock(3, 64)
            self.down_conv2 = ConvBlock(64, 128)
            self.down_conv3 = ConvBlock(128, 256)
            self.down_conv4 = ConvBlock(256, 512)
            self.down_conv5 = ConvBlock(512, 1024)

        else:
            vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            features = vgg.features

            self.down_conv1 = features[0:6]   
            self.down_conv2 = features[7:13]  
            self.down_conv3 = features[14:23] 
            self.down_conv4 = features[24:33] 
            
            # Stage 5 (Bottleneck): VGG16 chỉ có 512 channels. 
            # Ta lấy block cuối của VGG16 nối tiếp với 1 ConvBlock để nâng lên 1024 channels.
            # Bọc tất cả vào nn.Sequential và gán tên self.down_conv5 để giữ nguyên cấu trúc hàm forward.
            self.down_conv5 = nn.Sequential(features[34:43], ConvBlock(512, 1024))

        self.proj_conv = ConvBlock(1024, 1024)
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1))
        self.linear = nn.Linear(1024, self.num_class)
        
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_conv4 = ConvBlock(128, 64)

        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        down1 = self.down_conv1(x)
        max1 = self.maxpool(down1)

        down2 = self.down_conv2(max1)
        max2 = self.maxpool(down2)

        down3 = self.down_conv3(max2)
        max3 = self.maxpool(down3)

        down4 = self.down_conv4(max3)
        max4 = self.maxpool(down4)

        down5 = self.down_conv5(max4)
        b = x.shape[0]
        proj = self.proj_conv(self.maxpool(down5))
        proj = proj.contiguous().view(b, -1)
        cls = self.avg(down5).contiguous().view(b, -1)
        cls = self.linear(cls)


        up_t1 = self.up_transpose1(down5)

        up1 = self.up_conv1(torch.cat([down4, up_t1], dim=1))
        up_t2 = self.up_transpose2(up1)
        up2 = self.up_conv2(torch.cat([down3, up_t2], dim=1))
        up_t3 = self.up_transpose3(up2)
        up3 = self.up_conv3(torch.cat([down2, up_t3], dim=1))
        up_t4 = self.up_transpose4(up3)
        up4 = self.up_conv4(torch.cat([down1, up_t4], dim=1))

        out = torch.sigmoid(self.final(up4))

        if self.with_tsne_emb:
            return out, proj, cls, down5
        else:
            return out, proj, cls