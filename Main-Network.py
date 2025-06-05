import torch.nn as nn
import torch
from .quaternion_layers import QuaternionConv as QuaternionConv2d
import numpy as np

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QuaternionConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1

    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return QuaternionConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


class Bottleneck(nn.Module):


    def __init__(self, inplanes, planes, expansion=4, stride=1, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes // expansion) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)

        self.conv3 = conv3x3(width, width, stride)
        self.bn3 = norm_layer(width)

        self.conv4 = conv1x1(width, planes)
        self.bn4 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        out += identity
        out = self.relu(out)

        return out

class Quaternion_pyramid(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(Quaternion_pyramid, self).__init__()
        self.convT = QuaternionConv2d(in_channels//4, in_channels//4,kernel_size=3, padding=1)
        self.convT_dowdimension = QuaternionConv2d(in_channels, out_channels,kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
        self.bottleneck=Bottleneck(in_channels, in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid=nn.Sigmoid()
        #self.max_pool

    def forward(self, x):
        # 获取张量的形状
        batch_size, num_channels, height, width = x.size()
        # 初始化输出张量列表
        output_tensors = []
        x_split = torch.chunk(x, 2, dim=1)
        x1 =  x_split[0].clone()
        x11=  x_split[1].clone()
        x11_split = torch.chunk(x11, 2, dim=1)
        x2 =  x11_split[0].clone()
        x3 =  x11_split[1].clone()

        # 遍历每一组通道
        for group_idx in range(1, 4):
            scaled_channel_indices = [group_idx + 4 * i for i in range(num_channels // 16)]
            if group_idx==1:
                for idx in scaled_channel_indices:
                    # 将灰度值缩小为原来的四分之一
                    x1[:, idx, :, :] /= 4.0
            elif group_idx==2:
                for idx in scaled_channel_indices:
                    # 将灰度值缩小为原来的四分之一
                    x2[:, idx, :, :] /= 4.0
            else:
                for idx in scaled_channel_indices:
                    x3[:, idx, :, :] /= 4.0
            # 将处理后的张量添加到输出张量列表中
            #output_tensors.append(output_tensor)
    ##############################weight compution####
        x11 = torch.mean(x1, dim=1, keepdim=True)
        x12,_ = torch.max(x1, dim=1, keepdim=True)

        x21 = torch.mean(x2, dim=1, keepdim=True)
        x22,_ = torch.max(x2, dim=1, keepdim=True)

        x31 = torch.mean(x3, dim=1, keepdim=True)
        x32,_ = torch.max(x3, dim=1, keepdim=True)

        x1_weight=x11+x12
        x2_weight = x21 + x22
        x3_weight = x31 + x32

        attn1 = x1_weight @ x2_weight.transpose(2, 3) * (x1.shape[-1] ** -0.5)
        attn1 = self.sigmoid(attn1)

        attn2 = x1_weight @ x3_weight.transpose(2, 3) * (x1.shape[-1] ** -0.5)
        attn2 = self.sigmoid(attn2)

        attn3 = x2_weight @ x3_weight.transpose(2, 3) * (x1.shape[-1] ** -0.5)
        attn3 = self.sigmoid(attn3)
        ##############################weight compution####

        weight=(attn1+attn2+attn3)*0.33

        out=x*weight+self.bottleneck(x)
        out = self.convT_dowdimension(out)
        out = self.norm(out)
        return self.activation(out)

class ConvBatchNorm_quaternion(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm_quaternion, self).__init__()
        self.convT =QuaternionConv2d(in_channels, out_channels,kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.convT(x)
        return x

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv =Quaternion_pyramid(in_channels, out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
##################################
def _make_nConv1(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm1(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm1(out_channels, out_channels, activation))
    return nn.Sequential(*layers)
class ConvBatchNorm1(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm1, self).__init__()
        self.conv =QuaternionConv2d(in_channels, out_channels,kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class fusion_weight_compution(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self):
        super(fusion_weight_compution, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        t1 = x[:, 1:64:4, :, :]
        t2 = x[:, 2:64:4, :, :]
        t3 = x[:, 3:64:4, :, :]
        t4 = x[:, 4:65:4, :, :]

        avgout1 = torch.mean(t1, dim=1, keepdim=True)
        maxout1, _ = torch.max(t1, dim=1, keepdim=True)
        t1_weight = self.sigmoid(avgout1 + maxout1)

        avgout2 = torch.mean(t2, dim=1, keepdim=True)
        maxout2, _ = torch.max(t2, dim=1, keepdim=True)
        t2_weight = self.sigmoid(avgout2 + maxout2)

        avgout3 = torch.mean(t3, dim=1, keepdim=True)
        maxout3, _ = torch.max(t3, dim=1, keepdim=True)
        t3_weight = self.sigmoid(avgout3 + maxout3)

        avgout4 = torch.mean(t4, dim=1, keepdim=True)
        maxout4, _ = torch.max(t4, dim=1, keepdim=True)
        t4_weight = self.sigmoid(avgout4 + maxout4)

        weight_total = torch.cat([t1_weight, t2_weight, t3_weight, t4_weight], dim=1)
        return weight_total

class skipfeature_fusion(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self,out_channels):
        super(skipfeature_fusion, self).__init__()
        self.weight_compuion = fusion_weight_compution()
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()
    def forward(self, x1,x2):

        x1_weight=self.weight_compuion(x1)
        x2_weight=self.weight_compuion(x2)

        B,C,H,W=x2.shape

        result = torch.zeros(B, C, H, W, device=x2.device)
        result1 = torch.zeros(B, C, H, W, device=x2.device)

        for i in range(4):
            x2_channel_indices = torch.arange(i, C, step=4, device=x2.device)
            result[:, x2_channel_indices, :, :] = x1_weight[:, i, :, :].unsqueeze(1) * x2[:, x2_channel_indices, :, :]

            x1_channel_indices = torch.arange(i, C, step=4, device=x2.device)
            result1[:, x1_channel_indices, :, :] = x2_weight[:, i, :, :].unsqueeze(1) * x1[:, x1_channel_indices, :, :]

        # result=self.norm(result)
        # result1=self.norm(result1)
        x1 = torch.cat([result, result1], dim=1)

        return x1


######################################################
class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv1(in_channels, out_channels, nb_Conv, activation)
        self.fusion=skipfeature_fusion(in_channels//2)
    def forward(self, x, skip_x):
        out = self.up(x)
        x2 = self.fusion(skip_x, out).to(x.device)  # dim 1 is the channel dimension

        return self.nConvs(x2)

class UCTransNet(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,bilinear=True,img_size=224,vis=False):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.inc3 = ConvBatchNorm_quaternion(4, in_channels)

        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc =QuaternionConv2d(in_channels, 4, kernel_size=1, padding=0)
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        B,C,H,W=x.size()
        input_data = (B,1,H,W)
        params = nn.Parameter(torch.empty(*input_data)).to(x.device)
        nn.init.xavier_uniform_(params)
        T= torch.cat([params,x],dim=1)


        x1 = self.inc3(T)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.last_activation is not None:
            x=self.outc(x)
            x = x.sum(dim=1, keepdim=True)
            logits = self.last_activation(x)
            # print("111")
        else:
            logits = self.outc(x)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits


