import torch
import torch.nn as nn
import torch.nn.functional as F

class HoVerNetBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, x):
        return self.block(x)

class HoVerNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base_ch = 64

        # Encoder
        self.enc1 = self.conv_block(1, base_ch)
        self.enc2 = self.conv_block(base_ch, base_ch * 2)
        self.enc3 = self.conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = self.conv_block(base_ch * 4, base_ch * 8)

        # Decoder (shared for all branches)
        self.middle = self.conv_block(base_ch * 8, base_ch * 8)

        # Output branches
        self.np_branch = HoVerNetBranch(base_ch * 8, 1)           # Nuclei Pixel
        self.hv_branch = HoVerNetBranch(base_ch * 8, 2)           # Horizontal & Vertical maps
        self.tp_branch = HoVerNetBranch(base_ch * 8, num_classes) # Type prediction (optional)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x_middle = self.middle(x4)

        np_out = torch.sigmoid(self.np_branch(x_middle))
        hv_out = self.hv_branch(x_middle)
        tp_out = self.tp_branch(x_middle)

        return {"np": np_out, "hv": hv_out, "tp": tp_out}
