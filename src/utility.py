import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import lpips
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print("save:", args.save)
        print("load:", args.load)
        #exit()

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def save_patch(self, patch, epoch, is_best=False, idx=0):
        path = self.get_path('model/patches/')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(patch.state_dict(), self.get_path('model/patches/') + f"patch_{idx}.pt")
        #model.save(self.get_path('model/patches'), epoch, is_best=is_best, idx=idx)
    
    def save_everyepoch(self, trainer, epoch, is_best=False):
        save_path = self.get_path('model')+"/" + str(epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        trainer.model.save_every(save_path, epoch, is_best=is_best)

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(str(log) + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )
            print(filename)

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0
    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    if mse==0:
        return 1000
    return -10 * math.log10(mse)

lpips_values = []
loss_fn = lpips.LPIPS(net='alex')
def calc_lpips(sr, hr):
    global lpips_values
    sr.to(hr.device)
    
    for sr_img, hr_img in zip(sr, hr):
        lpips_value = loss_fn(sr_img.cpu(), hr_img.cpu())
        lpips_values.append(lpips_value.item())
    
    average_lpips = sum(lpips_values) / len(lpips_values)
    print("Average lpips:", average_lpips)
    return average_lpips

import skimage.metrics
ssim_values = []
def calc_ssim(sr, hr):
    global ssim_values
    for sr_img, hr_img in zip(sr, hr):
        sr_img_np = sr_img.permute(1, 2, 0).cpu().numpy()
        hr_img_np = hr_img.permute(1, 2, 0).cpu().numpy()
        ssim_value = skimage.metrics.structural_similarity(sr_img_np, hr_img_np, multichannel=True)
        ssim_values.append(ssim_value)
    
    average_ssim = sum(ssim_values) / len(ssim_values)
    print("Average SSIM:", average_ssim)
    return average_ssim

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    if args.cafm:
        if args.finetune:
            trainable = [{'params':[ param for name, param in target.named_parameters() if 'transformer' in name or 'gamma' in name]}]
        else:
            trainable = filter(lambda x: x.requires_grad, target.parameters())
    else:
        trainable = filter(lambda x: x.requires_grad, target.parameters())
        
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def make_patch_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    if args.cafm:
        if args.finetune:
            trainable = [{'params':[ param for name, param in target.named_parameters() if 'transformer' in name or 'gamma' in name]}]
        else:
            trainable = filter(lambda x: x.requires_grad, target.parameters())
    else:
        trainable = filter(lambda x: x.requires_grad, target.parameters())
        
    kwargs_optimizer = {'lr': args.patch_lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.patch_momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.patch_betas
        kwargs_optimizer['eps'] = args.patch_epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.patch_epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def make_trainChunk(num, length, segnum):
    chunk = int(int(length)//int(segnum))
    t_num = [[] for i in range(segnum)]
    for i in range(segnum):
        s = i*chunk + 1
        e = (i+1)*chunk
        t_num[i] = [j for j in range(len(num)) if s<=int(num[j])<=e]
    return t_num


def make_testChunk(length, filename, segnum=3, fps=30):
    if segnum == 3:
        if length == '45s':
            if int(filename)<=150 or 1351 <= int(filename) <= 1365:
                flag = 0
            elif 151 <= int(filename) <= 300 or 1366 <= int(filename) <= 1380:
                flag = 1
            elif 301 <= int(filename) <= 450 or 1381 <= int(filename) <= 1395:
                flag = 2
            elif 451 <= int(filename) <= 600 or 1396 <= int(filename) <= 1410:
                flag = 3
            elif 601 <= int(filename) <= 750 or 1411 <= int(filename) <= 1425:
                flag = 4
            elif 751 <= int(filename) <= 900 or 1426 <= int(filename) <= 1440:
                flag = 5
            elif 901 <= int(filename) <= 1050 or 1441 <= int(filename) <= 1455:
                flag = 6
            elif 1051 <= int(filename) <= 1200 or 1456 <= int(filename) <= 1470:
                flag = 7
            elif 1201 <= int(filename) <= 1350 or 1471 <= int(filename) <= 1485:
                flag = 8
            else:
                flag = 9
            return flag
        elif length =='15s':
            if 1 <= int(filename)<=150 or 451 <= int(filename) <= 465:
                flag = 0
            elif 151 <= int(filename) <= 300 or 466 <= int(filename) <= 480:
                flag = 1
            elif 301 <= int(filename) <= 450 or 481 <= int(filename) <= 495:
                flag = 2
            else:
                flag = 3 
            return flag
        elif length == '30s':
            if int(filename)<=150 or 901 <= int(filename) <= 915:
                flag = 0
            elif 151 <= int(filename) <= 300 or 916 <= int(filename) <= 930:
                flag = 1
            elif 301 <= int(filename) <= 450 or 931 <= int(filename) <= 945:
                flag = 2
            elif 451 <= int(filename) <= 600 or 946 <= int(filename) <= 960:
                flag = 3
            elif 601 <= int(filename) <= 750 or 961 <= int(filename) <= 975:
                flag = 4
            elif 751 <= int(filename) <= 900 or 976 <= int(filename) <= 990:
                flag = 5
            else:
                flag = 6 
            return flag
    else:
        if length == '15s':
            seg_size = int(int(15*fps)/segnum)
            test_seg_size = int(45/segnum)
            flag = None
            for i in range(segnum):
                if i*seg_size <= int(filename) <= (i+1)*seg_size or 450 + i*test_seg_size <= int(filename) <= 450 + (i+1)*test_seg_size:
                    flag = i
                    break
            if not flag: flag = segnum
            return flag
        if length == '45s':
            seg_size = int(int(45*fps)/segnum)
            test_seg_size = int(135/segnum)
            flag = None
            for i in range(segnum):
                if i*seg_size <= int(filename) <= (i+1)*seg_size or 1365 + i*test_seg_size <= int(filename) <= 1365 + (i+1)*test_seg_size:
                    flag = i
                    break
            if not flag: flag = segnum
            return flag
