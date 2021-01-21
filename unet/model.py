import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation_func=F.relu, batch_norm=True):
        super(DownsamplingBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation_func = activation_func
        self.batch_norm = batch_norm
        
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size)
        
        self.bnorm = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bnorm(x)
        x = self.activation_func(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bnorm(x)
        x = self.activation_func(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=True):
        super(UpsamplingBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm        
        self.upconv1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size)
        self.upconv2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, self.kernel_size)
        
    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, n_classes, first_out_channels=64):
        super(UNet, self).__init__()
        self.f_out_ch = first_out_channels

        # Downsampling
        self.downblock1 = DownsamplingBlock(in_channels=3, out_channels=self.f_out_ch, kernel_size=(3, 3))        
        self.downblock2 = DownsamplingBlock(in_channels=self.f_out_ch, out_channels=self.f_out_ch * 2, kernel_size=(3, 3))
        self.downblock3 = DownsamplingBlock(in_channels=self.f_out_ch * 2, out_channels=self.f_out_ch * 4, kernel_size=(3, 3))
        self.downblock4 = DownsamplingBlock(in_channels=self.f_out_ch * 4, out_channels=self.f_out_ch * 8, kernel_size=(3, 3))
        self.downblock5 = DownsamplingBlock(in_channels=self.f_out_ch * 8, out_channels=self.f_out_ch * 16, kernel_size=(3, 3))
        
        # Upsampling
        self.upblock1 = UpsamplingBlock(in_channels=self.f_out_ch * 16, out_channels=self.f_out_ch * 8, kernel_size=(3, 3))
        self.upblock2 = UpsamplingBlock(in_channels=self.f_out_ch * 8, out_channels=self.f_out_ch * 4, kernel_size=(3, 3))
        self.upblock3 = UpsamplingBlock(in_channels=self.f_out_ch * 4, out_channels=self.f_out_ch * 2, kernel_size=(3, 3))
        self.upblock4 = UpsamplingBlock(in_channels=self.f_out_ch * 2, out_channels=self.f_out_ch, kernel_size=(3, 3))
        
        self.downblock6 = DownsamplingBlock(in_channels=self.f_out_ch * 16, out_channels=self.f_out_ch * 8, kernel_size=(3, 3))        
        self.downblock7 = DownsamplingBlock(in_channels=self.f_out_ch * 8, out_channels=self.f_out_ch * 4, kernel_size=(3, 3))
        self.downblock8 = DownsamplingBlock(in_channels=self.f_out_ch * 4, out_channels=self.f_out_ch * 2, kernel_size=(3, 3))
        self.downblock9 = DownsamplingBlock(in_channels=self.f_out_ch * 2, out_channels=self.f_out_ch, kernel_size=(3, 3))
        
        self.output = nn.Conv2d(in_channels=self.f_out_ch, out_channels=n_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
    def crop(self, x1, x2):
        _, _, H, W = x2.shape
        x1_crop = torchvision.transforms.CenterCrop([H, W])(x1)
        return x1_crop

    def forward(self, x):
        c1 = self.downblock1(x)
        c1 = self.pool(c1)
        c2 = self.downblock2(c1)
        c2 = self.pool(c2)
        c3 = self.downblock3(c2)
        c3 = self.pool(c3)
        c4 = self.downblock4(c3)
        c4 = self.pool(c4)
        c5 = self.downblock5(c4)
        
        u6 = self.upblock1(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.downblock6(u6)
        
        u7 = self.upblock2(c6)
        c3 = self.crop(c3, u7)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.downblock7(u7)
        
        u8 = self.upblock3(c7)
        c2 = self.crop(c2, u8)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.downblock8(u8)
        
        u9 = self.upblock4(c8)
        c1 = self.crop(c1, u9)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.downblock9(u9)
        
        return self.output(c9)
