import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DynamicCon2dAttention(nn.Module):
    def __init__(self, channel_in, dynamic_number, delta):
        super(DynamicCon2dAttention, self).__init__()
        mid_channels = channel_in // 4
        self.delta = 1.0 / float(delta)
        self._module = nn.Sequential(
            nn.Linear(channel_in, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, dynamic_number),
        ).cuda()

    def forward(self, x):
        _x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze().cuda()
        #print(x.size(), _x.size(), _x.squeeze().size())
        _x_ = self._module(_x)
        #print(_x_.size(), self.delta)
        y = _x_ * self.delta
        return F.softmax(y).cuda()

class ConvDY2d(nn.Module):
    """
        DynamicConv2d from paper "Dynamic Convolution: Attention over Convolution Kernels"
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, dynamic_number = 4, delta = 30):
        super(ConvDY2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dynamic_number = dynamic_number
        # for softmax ratio
        self.delta = delta

        self.attention_module = DynamicCon2dAttention(self.in_channels, self.dynamic_number, self.delta)
        self.dynamic_weights = torch.nn.Parameter(torch.randn([(self.dynamic_number), self.out_channels, self.in_channels, self.kernel_size, self.kernel_size]), requires_grad = True)
        if bias:
            self.dynamic_bias = torch.nn.Parameter(torch.randn([(self.dynamic_number), self.out_channels]), requires_grad = True)
        else:
            self.dynamic_bias = None

    def forward(self, x):
        # get the softmax weights
        softmax_weights = self.attention_module(x)

        # caculate the final_weights, final_bias
        for i in range(x.size()[0]):
            for j in range(int(self.dynamic_number)):
                if j == 0:
                    bias = None
                    if x.size()[0] == 1:
                        weights = softmax_weights[j].cuda() * self.dynamic_weights[j].cuda()
                        if type(self.dynamic_bias) == torch.Tensor:
                            bias = softmax_weights[j].cuda() * self.dynamic_bias[j].cuda()
                    else:
                        weights = softmax_weights[i, j].cuda() * self.dynamic_weights[j].cuda()
                        if type(self.dynamic_bias) == torch.Tensor:
                            bias = softmax_weights[i, j].cuda() * self.dynamic_bias[j].cuda()
                else:
                    if x.size()[0] == 1:
                        weights = torch.add(weights.cuda(), softmax_weights[j].cuda() * self.dynamic_weights[j].cuda())
                        if type(self.dynamic_bias) == torch.Tensor:
                            bias = torch.add(bias.cuda(), softmax_weights[j].cuda() * self.dynamic_bias[j].cuda())
                    else:
                        weights = torch.add(weights.cuda(), softmax_weights[i, j].cuda() * self.dynamic_weights[j].cuda())
                        if type(self.dynamic_bias) == torch.Tensor:
                            bias = torch.add(bias.cuda(), softmax_weights[i, j].cuda() * self.dynamic_bias[j].cuda())
            single_out = F.conv2d(x[i, :, :, :].unsqueeze(0), weight = weights, bias = bias, stride = self.stride, padding = self.padding, dilation = self.dilation, groups = self.groups)
            if i == 0:
                out = torch.zeros([x.size()[0]] + list(single_out.squeeze().size()))
            out[i, :, :, :] = single_out
        return out.cuda()