import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_no_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_no_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
    )

def conv_no_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=int(inp/8), bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

def conv_no_bn_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        #input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        #baseline
        #self.stage1_1 = nn.Sequential(
        #    conv_bn(3, 8, 2, leaky = 0.1),    # 3
        #    conv_dw(8, 16, 2),   # 7
        #    conv_dw(16, 32, 2),  # 11
        #    conv_dw(32, 32, 2),  # 19
        #)
        #self.stage1_2 = nn.Sequential(
        #    conv_dw(32, 64, 2),  # 27
        #    conv_dw(64, 64, 1),  # 43
        #)
        #self.stage2 = nn.Sequential(
        #    conv_dw(64, 128, 2),  # 43 + 16 = 59
        #    conv_dw(128, 128, 1), # 59 + 32 = 91
        #    conv_dw(128, 128, 1), # 91 + 32 = 123
        #    conv_dw(128, 128, 1), # 123 + 32 = 155
        #    conv_dw(128, 128, 1), # 155 + 32 = 187
        #    conv_dw(128, 128, 1), # 187 + 32 = 219
        #)
        #self.stage3 = nn.Sequential(
        #    conv_dw(128, 256, 2), # 219 +3 2 = 241
        #    conv_dw(256, 256, 1), # 241 + 64 = 301
        #)
        
        #no stride
        self.stage1_1 = nn.Sequential(
            conv_bn(3, 16, 2, leaky = 0.1),    # 3
            conv_dw(16, 32, 2),   # 7
            conv_dw(32, 64, 2),  # 11
            conv_dw(64, 64, 1),  # 19
        )
        self.stage1_2 = nn.Sequential(
            conv_dw(64, 64, 2)
        )
        self.stage1_3 = nn.Sequential(
            conv_dw(64, 16, 1),
        )
        self.stage1_s = nn.Sequential(
            conv_dw(64, 128, 2)
        )
        
        self.stage2_1 = nn.Sequential(
            conv_dw(16, 128, 2)  # 43 + 16 = 59
        )
        self.stage2_2 = nn.Sequential(
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1)
        )
        self.stage2_3 = nn.Sequential(
            conv_dw(128, 32, 1)
        )
        self.stage2_s = nn.Sequential(
            conv_dw(128, 256, 2)
        )
        
        self.stage3_1 = nn.Sequential(
            conv_dw(32, 256, 2)
        )
        self.stage3_2 = nn.Sequential(
            conv_dw(256, 64, 1)
        )

    def forward(self, x):
        x1_1 = self.stage1_1(x)
        x1_2 = self.stage1_2(x1_1)
        x1_3 = self.stage1_3(x1_2)
        x1 = F.interpolate(x1_3, size=[x1_1.size(2), x1_1.size(3)], mode="nearest")
        x1_s = self.stage1_s(x1_2)
    
        x2_1 = self.stage2_1(x1_3) + x1_s
        x2_2 = self.stage2_2(x2_1)
        x2_3 = self.stage2_3(x2_2)
        x2 = F.interpolate(x2_3, size=[x1_2.size(2), x1_2.size(3)], mode="nearest")
        x2_s = self.stage2_s(x2_2)
    
        x3_1 = self.stage3_1(x2_3) + x2_s
        x3_2 = self.stage3_2(x3_1)
        x3 = F.interpolate(x3_2, size=[x2_2.size(2), x2_2.size(3)], mode="nearest")
        
        return [x1, x2, x3]

    #def forward(self, x):
    #    x1_1 = self.stage1_1(x)
    #    x1_2 = self.stage1_2(x1_1)
    #    x2_1 = self.stage2(x1_2)    
    #    x3_1 = self.stage3(x2_1)
    #    
    #    return [x1_2, x2_1, x3_1]

class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes, st=1):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, int(expand_planes/2), kernel_size=1, stride=st)
        self.conv3 = nn.Conv2d(squeeze_planes, int(expand_planes/2), kernel_size=3, stride=st, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #        m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out2 = self.conv3(x)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class SqueezeNetV1(nn.Module):
    def __init__(self):
        super(SqueezeNetV1, self).__init__()
        #self.stage1_1 = nn.Sequential(
        #    conv_no_bn(3, 4, 2, leaky = 0.1),
        #    fire(4, 8, 8, 2),   
        #    fire(8, 16, 16, 2), 
        #    fire(16, 32, 32, 2),
        #)
        #self.stage1_2 = nn.Sequential(
        #    fire(32, 32, 32, 2),
        #)
        #
        #self.stage2 = nn.Sequential(
        #    fire(32, 32, 64, 2),
        #    fire(64, 32, 64, 1),
        #    fire(64, 32, 64, 1),
        #    fire(64, 32, 64, 1),
        #)
        #self.stage3 = nn.Sequential(
        #    fire(64, 64, 128, 2), 
        #    fire(128, 64, 128, 1),
        #    fire(128, 64, 128, 1),
        #    fire(128, 64, 128, 1),
        #)
        
        #no stride
        self.stage1_1 = nn.Sequential(
            conv_no_bn(3, 4, 2, leaky = 0.1),
            fire(4, 8, 8, 2),   
            fire(8, 16, 16, 2), 
        )
        self.stage1_2 = nn.Sequential(
            fire(16, 32, 32, 2),
            fire(32, 32, 32, 1),
        )
        
        self.stage2 = nn.Sequential(
            fire(32, 32, 64, 2),
            fire(64, 32, 64, 1),
            fire(64, 32, 64, 1),
            fire(64, 32, 64, 1),
        )
        self.stage3 = nn.Sequential(
            fire(64, 64, 128, 2), 
            fire(128, 64, 128, 1),
            fire(128, 64, 128, 1),
            fire(128, 64, 128, 1),
        )

    def forward(self, x):
        x1_1 = self.stage1_1(x)
        x1_2 = self.stage1_2(x1_1)
        x1 = F.interpolate(x1_2, size=[x1_1.size(2), x1_1.size(3)], mode="nearest")
    
        x2_1 = self.stage2(x1_2)
        x2 = F.interpolate(x2_1, size=[x1_2.size(2), x1_2.size(3)], mode="nearest")
    
        x3_1 = self.stage3(x2_1)
        x3 = F.interpolate(x3_1, size=[x2_1.size(2), x2_1.size(3)], mode="nearest")
        
        return [x1, x2, x3]

    #def forward(self, x):
    #    x1_1 = self.stage1_1(x)
    #    x1_2 = self.stage1_2(x1_1)
    #    #x1 = F.interpolate(x1_2, size=[x1_1.size(2), x1_1.size(3)], mode="nearest")
    #
    #    x2_1 = self.stage2(x1_2)
    #    #x2 = F.interpolate(x2_1, size=[x1_2.size(2), x1_2.size(3)], mode="nearest")
    #
    #    x3_1 = self.stage3(x2_1)
    #    #x3 = F.interpolate(x3_1, size=[x2_1.size(2), x2_1.size(3)], mode="nearest")
    #    
    #    return [x1_2, x2_1, x3_1]

class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        #baseline
        #self.stage1_1 = nn.Sequential(
        #    conv_no_bn(3, 4, 2, leaky = 0.1),
        #    fire(4, 8, 8, 2),   
        #    fire(8, 16, 16, 2)
        #)
        #self.stage1_2 = nn.Sequential(        
        #    fire(16, 32, 32, 2)
        #)
        #self.stage1_3 = nn.Sequential(
        #    fire(32, 32, 32, 2)
        #)
        #
        #self.stage2 = nn.Sequential(
        #    fire(32, 32, 64, 2),
        #    fire(64, 32, 64, 1),
        #    fire(64, 32, 64, 1),
        #    fire(64, 32, 64, 1)
        #)
        #self.stage3 = nn.Sequential(
        #    fire(64, 64, 128, 2), 
        #    fire(128, 64, 128, 1),
        #    fire(128, 64, 128, 1),
        #    fire(128, 64, 128, 1)
        #)
        
        #half
        #self.stage1_1 = nn.Sequential(
        #    conv_no_bn(3, 4, 2, leaky = 0.1),
        #    fire(4, 8, 8, 2),   
        #    fire(8, 8, 8, 2)
        #)
        #self.stage1_2 = nn.Sequential(        
        #    fire(8, 16, 16, 2)
        #)
        #self.stage1_3 = nn.Sequential(
        #    fire(16, 16, 16, 2)
        #)
        #
        #self.stage2 = nn.Sequential(
        #    fire(16, 16, 32, 2),
        #    fire(32, 16, 32, 1),
        #    fire(32, 16, 32, 1),
        #    fire(32, 16, 32, 1)
        #)
        #self.stage3 = nn.Sequential(
        #    fire(32, 32, 64, 2), 
        #    fire(64, 32, 64, 1),
        #    fire(64, 32, 64, 1),
        #    fire(64, 32, 64, 1)
        #)
        
        #double
        self.stage1_1 = nn.Sequential(
            conv_no_bn(3, 8, 2, leaky = 0.1),
            fire(8, 16, 16, 2),   
            fire(16, 16, 32, 2)
        )
        self.stage1_2 = nn.Sequential(        
            fire(32, 32, 64, 2)
        )
        self.stage1_3 = nn.Sequential(
            fire(64, 32, 64, 2)
        )
        
        self.stage2 = nn.Sequential(
            fire(64, 64, 128, 2),
            fire(128, 64, 128, 1),
            fire(128, 64, 128, 1),
            fire(128, 64, 128, 1)
        )
        self.stage3 = nn.Sequential(
            fire(128, 128, 256, 2), 
            fire(256, 128, 256, 1),
            fire(256, 128, 256, 1),
            fire(256, 128, 256, 1)
        )

    def forward(self, x):
        x1_1 = self.stage1_1(x)
        x1_2 = self.stage1_2(x1_1)
        x1_3 = self.stage1_3(x1_2)
        x1 = F.interpolate(x1_3, size=[x1_2.size(2), x1_2.size(3)], mode="nearest")

        x2_1 = self.stage2(x1_3)
        x2 = F.interpolate(x2_1, size=[x1_3.size(2), x1_3.size(3)], mode="nearest")

        x3_1 = self.stage3(x2_1)
        x3 = F.interpolate(x3_1, size=[x2_1.size(2), x2_1.size(3)], mode="nearest")
        
        return [x1, x2, x3, x1_1]