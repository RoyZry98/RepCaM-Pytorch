import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

import utility
import data
import model
import loss
from option import args
from trainer_cafm import Trainer_cafm

import pdb
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def get_model_size(model):
    param_count = sum(p.numel() for p in model.parameters())
    
    param_size = param_count * 4 / (1024 * 1024)
    
    return {
        'parameters': param_count,
        'size_mb': param_size
    }

def print_param_info(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}")
        print(f"Type: {param.dtype}")
        print(f"Shape: {param.shape}")
        print("-" * 50)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            #pdb.set_trace()
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            print(get_model_size(_model))
            if args.cafm:
                t = Trainer_cafm(args, loader, _model, _loss, checkpoint)
            else:
                # print("u have to enter --cafm in command")
                # assert(0)
                t = Trainer_cafm(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()
            checkpoint.done()

if __name__ == '__main__':
    # U can change the random seed by yourself
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    main()
