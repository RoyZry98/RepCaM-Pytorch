import torch.nn as nn
#from mmcv.cnn.bricks import PLUGIN_LAYERS
import torch


class FixedPatchPrompter_image(nn.Module):
    def __init__(self, prompt_size = 12, std = 1):
        super(FixedPatchPrompter_image, self).__init__()
        self.psize = prompt_size
        self.patch = nn.Parameter(std * torch.randn([3, self.psize, self.psize]))

    def forward(self, x):
        isize = x.shape[-2:]
        prompt = torch.zeros([x.shape[0], 3, isize[0], isize[1]], device='cuda')
        prompt[:, :, :self.psize, :self.psize] = self.patch.unsqueeze(0)
        return x + prompt
    

# # for feature level  
#@PLUGIN_LAYERS.register_module()
class FixedPatchPrompter_feature_1(nn.Module):
    def __init__(self, prompt_size = 24, std = 1):
        super(FixedPatchPrompter_feature_1, self).__init__()
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([64, prompt_size, prompt_size])*std) #for feature size of espcn_lf

    def forward(self, x):
        tmp = torch.zeros_like(x)
        #print(tmp.shape, x.shape)
        tmp[:,:, :self.psize, :self.psize] = self.patch.unsqueeze(0)
        
        return x + tmp
    
class FixedPatchPrompter_feature_default(nn.Module):
    def __init__(self, prompt_size, image_size):
        super(FixedPatchPrompter_feature_default, self).__init__()
        self.isize = image_size
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([2, 2048, self.psize, self.psize])) #2 is batchsize, 2048 is feature dimension

    def forward(self, x):
        tmp = torch.zeros_like(x)
        tmp[:,:, :self.psize, :self.psize] = self.patch
        return x + tmp