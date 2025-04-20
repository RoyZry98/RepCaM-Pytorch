import math
from torch import nn
import torch.nn.init as init
from .common import ContentAwareFM
import torch

from .fix_patch_prompt import FixedPatchPrompter_feature_1

def make_model(args, parent=False):
    return ESPCN(args)


def set_padding_size(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class ContentAwareFM(nn.Module):
    # hello ckx
    def __init__(self, in_channel, kernel_size):

        super(ContentAwareFM, self).__init__()
        padding = set_padding_size(kernel_size, 1)
        self.transformer = nn.Conv2d(in_channel, in_channel, kernel_size,
                                     padding=padding, groups=in_channel//2)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
    def forward(self, x):
        return self.transformer(x) * self.gamma + x



class ESPCN(nn.Module):
    def __init__(self, args):
        super(ESPCN, self).__init__()
        # self.act_func = nn.LeakyReLU(negative_slope=0.2)
        self.act_func = nn.ReLU(inplace=True)
        self.scale = int(args.scale[0])  # use scale[0]
        self.n_colors = args.n_colors
        self.cafm = args.cafm
        self.use_cafm = args.use_cafm
        self.segnum = args.segnum
        #print(args.scale)
        self.prompter = FixedPatchPrompter_feature_1(args.patch_size // self.scale, std=args.std)
        # conv1

        self.conv1 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))

        self.conv1_0_0 = nn.Conv2d(self.n_colors, 64, (1, 1), (1, 1), (0, 0))
        self.conv1_0_1 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv1_0_2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        self.conv1_1_0 = nn.Conv2d(self.n_colors, 64, (1, 1), (1, 1), (0, 0))
        self.conv1_1_1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        self.conv1_2_0 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))


        #conv2
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv2_0_0 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv2_0_1 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv2_0_2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        self.conv2_1_0 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv2_1_1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        self.conv2_2_0 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        self.conv2_3_0 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv2_3_1 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv2_3_2 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv2_3_3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)) 

        #conv3
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3_0_0 = nn.Conv2d(64, 32, (1, 1), (1, 1), (0, 0))
        self.conv3_0_1 = nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0))
        self.conv3_0_2 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))

        self.conv3_1_0 = nn.Conv2d(64, 32, (1, 1), (1, 1), (0, 0))
        self.conv3_1_1 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))

        self.conv3_2_0 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))

        self.conv3_3_0 = nn.Conv2d(64, 32, (1, 1), (1, 1), (0, 0))
        self.conv3_3_1 = nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0))
        self.conv3_3_2 = nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0))
        self.conv3_3_3 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))

        self.conv4 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
        self.conv4_0_0 = nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0))
        self.conv4_0_1 = nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0))
        self.conv4_0_2 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))

        self.conv4_1_0 = nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0))
        self.conv4_1_1 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))

        self.conv4_2_0 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))

        self.pixel_shuffle = nn.PixelShuffle(self.scale)
        
        # self._initialize_weights()

        if self.cafm:
            if self.use_cafm:
                self.cafms1 = nn.ModuleList([ContentAwareFM(64,1) for _ in range(self.segnum)])
                self.cafms2 = nn.ModuleList([ContentAwareFM(64,1) for _ in range(self.segnum)])
                self.cafms3 = nn.ModuleList([ContentAwareFM(32,1) for _ in range(self.segnum)])


    def forward(self, x, num):
        if self.cafm:
            out = self.act_func(self.conv1(x))
            if self.use_cafm:
                out = self.cafms1[num](out)
            out = self.act_func(self.conv2(out))
            if self.use_cafm:
                out = self.cafms2[num](out)
            out = self.act_func(self.conv3(out))
            if self.use_cafm:
                out = self.cafms3[num](out)
            out = self.pixel_shuffle(self.conv4(out))
            return out
        else:
            out0 = self.act_func(self.conv1_0_2(self.conv1_0_1(self.conv1_0_0(x))) + self.conv1_1_1(self.conv1_1_0(x)) + self.conv1_2_0(x))
            #print(out0.shape)
            out0 = self.prompter(out0)
            out1 = self.act_func(self.conv2_0_2(self.conv2_0_1(self.conv2_0_0(out0))) + self.conv2_1_1(self.conv2_1_0(out0)) + self.conv2_2_0(out0))
            # out1 = self.act_func(self.conv2_0_2(self.conv2_0_1(self.conv2_0_0(out0))) + self.conv2_1_1(self.conv2_1_0(out0)) + self.conv2_2_0(out0))
            # out1 = self.act_func(self.conv2(out0))

            out2 = self.act_func(self.conv3_0_2(self.conv3_0_1(self.conv3_0_0(out1))) + self.conv3_1_1(self.conv3_1_0(out1)) + self.conv3_2_0(out1))
            # out2 = self.act_func(self.conv3_0_2(self.conv3_0_1(self.conv3_0_0(out1))) + self.conv3_1_1(self.conv3_1_0(out1)) + self.conv3_2_0(out1))
            # out2 = self.act_func(self.conv3(out1))
            out3 = self.pixel_shuffle(self.conv4_0_2(self.conv4_0_1(self.conv4_0_0(out2))) + self.conv4_1_1(self.conv4_1_0(out2)) + self.conv4_2_0(out2))
            
            return out3 

            
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)