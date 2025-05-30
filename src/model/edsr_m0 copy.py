import numpy as np
import torch
import torch.nn as nn

from model import common_m0
#from fix_patch_prompt import FixedPatchPrompter_image, FixedPatchPrompter_feature

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}
import pdb


def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common_m0.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        self.numbers = n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        self.cafm = args.cafm
        self.n_resblocks = args.n_resblocks
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common_m0.MeanShift(args.rgb_range)
        self.add_mean = common_m0.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        if args.cafm:
            m_body = [common_m0.ResBlock(conv, n_feats, kernel_size, args, bn=True, act=act, res_scale=args.res_scale) for _ in range(n_resblocks)]
        else:
            m_body = [common_m0.ResBlock_org(conv, n_feats, kernel_size, args, act=act, res_scale=args.res_scale) for _ in range(n_resblocks)]
        m_body.append(common_m0.RepBlock_m0(conv, n_feats, kernel_size, args, bias=True ))
        # define tail module
        m_tail = [
            common_m0.Upsampler(conv, scale, n_feats, kernel_size, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        if args.cafm:
            self.body = nn.ModuleList(m_body)
        else:
            self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, num):
        x = self.sub_mean(x)
        x = self.head(x)
        #cafm
        if self.cafm:
            res = x
            for i in range(self.numbers):
                res = self.body[i](res, num)
            res = self.body[self.n_resblocks](res)
            res += x
        #original
        else:
            res = self.body(x)
            res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
            # CaFM change the weight model name
            else:
                #print(name)
                name = name.replace("2.weight","3.weight") if "weight" in name else name.replace("2.bias","3.bias")
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)