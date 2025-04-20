import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
import sys
import numpy as np
import cv2 as cv
import imageio
from model.fix_patch_prompt import FixedPatchPrompter_image

class Trainer_cafm():
    def __init__(self, args, loader, my_model, my_loss, ckp: utility.checkpoint):
        self.args = args
        self.scale = args.scale
        
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        self.patch = FixedPatchPrompter_image(prompt_size = args.prompt_size, std = args.std).cuda()
        self.patch_optimizer = utility.make_patch_optimizer(args, self.patch)

        if args.use_cafm:
            self.patch = [FixedPatchPrompter_image(prompt_size = args.prompt_size, std = args.std).cuda() 
                          for i in range(args.segnum + 1)]
            self.patch_optimizer = [utility.make_patch_optimizer(args, self.patch[i]) 
                                    for i in range(args.segnum + 1)]

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        
        if self.args.use_cafm:
            if args.patch_load != '':
                print()
                print(args.patch_load)
                path = args.patch_load
                for i in range(args.segnum + 1):
                    patch_checkpoint_path = os.path.join(path, f'patch_{i}.pt')
                    if os.path.exists(patch_checkpoint_path):
                        checkpoint = torch.load(patch_checkpoint_path)
                        self.patch[i].load_state_dict(checkpoint)
                        #self.patch_optimizer[i].load_state_dict(checkpoint['optimizer'])
                        print(f"patch{i} loaded")
                    else:
                        print(patch_checkpoint_path + " not exists")

        self.error_last = 1e8

    def train(self):

        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        plr = self.patch_optimizer[0].get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.ckp.write_log(
            '[Epoch {}]\tPatch learning rate: {:.2e}'.format(epoch, Decimal(plr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        length = self.args.data_range.split('/')[0].split('-')[1]
        segnum = self.args.segnum
        for batch, (lr, hr, num,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            if self.args.use_cafm:
                for opt in self.patch_optimizer:
                    opt.zero_grad()
            #use_cafm
            if self.args.use_cafm:
                t_num = utility.make_trainChunk(num, length, segnum)
                numlist = [(t_num[i],i)for i in range(segnum)]
                #print(numlist)
                loss = 0
                for i in numlist:
                    if len(i[0])!=0:
                        pre_sr = self.patch[i[1]](lr[i[0]])
                        sr = self.model(pre_sr, 0, i[1])
                        loss += self.loss(sr, hr[i[0]])*len(i[0])
                loss = loss/len(num)
            elif self.args.chunked:
                sr = self.model(lr, 0, np.array(num).astype(int))
                loss = self.loss(sr, hr)
            #baseline
            else:
                sr = self.model(self.patch(lr), 0, num)
                loss = self.loss(sr, hr)

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            if self.args.use_cafm:
                for opt in self.patch_optimizer:
                    opt.step()
            else:
                self.patch_optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{:.1f}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    #self.loss.display_loss(batch),
                    loss.data,
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
            print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
            print(f"Reserved memory: {torch.cuda.memory_reserved()} bytes")

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        if self.args.use_cafm:
            for opt in self.patch_optimizer:
                opt.schedule()
        else:
            self.patch_optimizer.step()

    def test(self):


        # low_hr_path = '/home/dlx/CaFM-Pytorch-ICCV2021-main/src/new_data/new_data/low_hr_7000/'
        # low_list = os.listdir(low_hr_path)
        # low_list.sort(key= lambda x:int(x[:-4]))
        # # print( low_list)

        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
            
                k = 0

                for lr, hr, filename in tqdm(d, ncols=80):
                    # print(filename)
                    lr, hr = self.prepare(lr, hr)
                    filename[0] = filename[0].split('x')[0]
                    if self.args.is45s:
                        flag = utility.make_testChunk('45s', filename[0])
                    elif self.args.is15s:
                        flag = utility.make_testChunk('15s', filename[0])
                    elif self.args.is30s:
                        flag = utility.make_testChunk('30s', filename=[0])
                    
                    if self.args.use_cafm:
                        sr = self.model(self.patch[flag](lr), idx_scale, flag)
                    else:
                        sr = self.model(self.patch(lr), idx_scale, flag)
                    # print(sr.shape)



                    # # low_hr = cv.imread(low_hr_path+ low_list[k])
                    
                    # # low_hr = low_hr[:, :, ::-1]
                    # # print(low_hr_path+ low_list[k])
                    # low_hr = cv.imread(low_hr_path + low_list[k], flags=cv.IMREAD_COLOR)
                    
                    # low_hr = cv.cvtColor(low_hr,cv.COLOR_BGR2RGB)
                    # print(low_hr_path+ low_list[k],filename)

                    # k = k+1
                    # device = hr.device
                    # low_hr = torch.from_numpy(low_hr).to(device)
                    # # print(low_hr[0,0,0])
                    # low_hr = low_hr.permute(2,0,1).unsqueeze(dim=0)
                   
                    # # print(low_hr.shape)



                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]

                    #utility.calc_lpips(sr, hr)
                    #utility.calc_ssim(sr, hr)

                    # save_list = [low_hr]

                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results: 
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if not self.args.use_cafm: 
            self.ckp.write_log(f"Mean of the patch prompt: {torch.mean(self.patch.patch)}, Std: {torch.std(self.patch.patch)}")
        else:
            for idx, patch in enumerate(self.patch):
                self.ckp.write_log(f"Mean of the patch prompt seg {idx}: {torch.mean(patch.patch)}, Std: {torch.std(patch.patch)}")

        
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
            #self.ckp.save_everyepoch(self, epoch, is_best=True)
            if best[1][0, 0] + 1 == epoch:
                for i, model in enumerate(self.patch):
                    self.ckp.save_patch(model, epoch, is_best=True, idx=i)
            elif epoch % 40 == 0:
                for i, model in enumerate(self.patch):
                    self.ckp.save_patch(model, epoch, is_best=True, idx=f"epo_{epoch}_{i}")


        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

