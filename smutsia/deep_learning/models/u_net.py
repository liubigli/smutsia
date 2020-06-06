import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, nb0_filters=32, kernel_size=3, max_pooling=2):
        """
        Parameters
        ----------
        n_channels: int

        n_classes: int

        bilinear: pool

        nb0_filters: int

        kernel_size: int or tuple

        max_pooling: int or tuple
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.downs = []
        self.ups = []
        self.kernel_size = kernel_size
        self.max_pooling = max_pooling
        self.inc = DoubleConv(n_channels, nb0_filters, kernel_size=self.kernel_size)
        factor = 2 if bilinear else 1

        for i in range(4):
            if i == 3:
                self.downs.append(
                    Down(nb0_filters * (2 ** i), nb0_filters * (2**(i+1)) // factor, max_pool=self.max_pooling)
                )
                self.ups.append(
                    Up(nb0_filters * 2, nb0_filters, bilinear, scale_factor=self.max_pooling)
                )
            else:
                self.downs.append(
                    Down(nb0_filters * (2 ** i), nb0_filters * (2**(i+1)), max_pool=self.max_pooling)
                )
                self.ups.append(
                    Up(nb0_filters * (2 ** (4 - i)), nb0_filters * (2 ** (3 - i)) // factor, bilinear,
                       scale_factor=self.max_pooling)
                )

        self.outc = OutConv(nb0_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        down_x = [x1]
        print(x1.shape)
        # encoder
        for i in range(4):
            down_x.append(self.downs[i](down_x[i]))
        x_up = down_x[-1]
        # decoder
        for i in range(4):
            # we need to stack the out put of previus layter with the corresponding layer in downs
            x_up = self.ups[i](x_up, down_x[3-i])

        logits = self.outc(x_up)

        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLu) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mid_channels is None:
            self.mid_channels = out_channels
        else:
            self.mid_channels = mid_channels

        self.kernel_size = kernel_size

        self.double_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=2):
        super(Down, self).__init__()
        self.max_pool = nn.Sequential(nn.MaxPool2d(max_pool),
                                      DoubleConv(in_channels=in_channels, out_channels=out_channels)
                                      )

    def forward(self, x):
        return self.max_pool(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=in_channels // 2, kernel_size=2, stride=scale_factor)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    u_net = UNet(n_channels=6, n_classes=1, max_pooling=(1, 2), bilinear=False)
    xinput = torch.rand([5, 6, 64, 2048])
    out = u_net.forward(xinput)
    from smutsia.utils.torchsummary import summary
    summary(u_net, input_size=(6, 64, 2048), device='cpu')
    print(out.shape)
