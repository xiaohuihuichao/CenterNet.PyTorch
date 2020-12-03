import torch.nn as nn


class CBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.layer(x)
    
    
# b 是小的 feature map
def transConv_add_merge(deconv_layer, a, b, merge_layer):
    b = deconv_layer(b)
    return merge_layer(b+a)
    
class FPN(nn.Module):
    def __init__(self, channels_list, out_channel):
        super().__init__()
        
        self.conv1 = CBA(channels_list[0], out_channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = CBA(channels_list[1], out_channel, kernel_size=1, stride=1, padding=0)
        self.conv3 = CBA(channels_list[2], out_channel, kernel_size=1, stride=1, padding=0)
        
        self.conv_trans3 = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=2, stride=2)
        self.conv_trans2 = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=2, stride=2)
        
        self.merge2 = CBA(out_channel, out_channel, kernel_size=3, padding=1, stride=1)
        self.merge1 = CBA(out_channel, out_channel, kernel_size=3, padding=1, stride=1)
        
    def forward(self, x):
        x1, x2, x3 = x
        
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        
        out = transConv_add_merge(self.conv_trans3, x2, x3, self.merge2)
        out = transConv_add_merge(self.conv_trans2, x1, out, self.merge1)
        return out
        
        
if __name__ == "__main__":
    import torch
    channels_list = [10, 20, 30]
    x1 = torch.rand(1, channels_list[0], 400, 400)
    x2 = torch.rand(1, channels_list[1], 200, 200)
    x3 = torch.rand(1, channels_list[2], 100, 100)
    
    fpn = FPN(channels_list, 50)
    y = fpn([x1, x2, x3])
    print(y.shape)
    