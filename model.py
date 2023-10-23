# -*- coding: utf-8 -*-
import functools
import math
# from typing_extensions import final

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import init
import cv2
from utils import init_weights
EPS = 1e-12
bias = True


class ResBlock_dense(nn.Module):
    def __init__(self, fin=128, fout=128):
        super(ResBlock_dense, self).__init__()
        self.conv0 = nn.Conv2d(fin, fout, 3, 1, 1, bias=bias)
        self.conv1 = nn.Conv2d(fout, fout, 3, 1, 1, bias=bias)
    def forward(self, x):
        # x[0]: concat; x[1]: fea
        fea = F.leaky_relu(self.conv0(x[0]), 0.2, inplace=True)
        fea = F.leaky_relu(self.conv1(fea), 0.2, inplace=True)
        fea = fea * (x[2] + 1)
        x0 = x[1] + fea
        return (x0, x[1], x[2])

class ResBlock_conv_dense(nn.Module):
    def __init__(self, fin=128, fout=128):
        super(ResBlock_conv_dense, self).__init__()
        self.conv0 = nn.Conv2d(fin, fout, 3, 1, 1, bias=bias)
        self.conv1 = nn.Conv2d(fout, fout, 3, 1, 1, bias=bias)
    def forward(self, x):
        fea = F.leaky_relu(self.conv0(x[0]), 0.2, inplace=True)
        fea = F.leaky_relu(self.conv1(fea), 0.2, inplace=True)
        x0 = x[1] + fea
        return (x0, x[1], x[2])


class CreateGenerator(nn.Module):
    def __init__(self, inchannels=8, grads=3):
        super(CreateGenerator, self).__init__()
        f = 32
        # ms and pan images
        self.encoder21 = nn.Sequential(nn.Conv2d(inchannels, f, 3, 1, 1, bias=bias),
                                       nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 1, 1, bias=bias))
        self.encoder22 = nn.Sequential(nn.Conv2d(1, f, 3, 1, 1, bias=bias),
                                       nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 1, 1, bias=bias))

        self.encoder41 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 2, 1, bias=bias))
        self.encoder42 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 2, 1, bias=bias))

        self.encoder31 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 2, 1, bias=bias))
        self.encoder32 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 2, 1, bias=bias))

        self.conv0 = nn.Conv2d(f * 2, f * 2, 3, 1, 1, bias=bias)

        self.CondNet = nn.Sequential(nn.Conv2d(grads, f, 3, 1, 1, bias=bias),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(f, f, 3, 1, 1, bias=bias),
                                    nn.LeakyReLU(0.2, True))
        
        self.cond_stage1 = nn.Sequential(nn.Conv2d(f, f, 4, 2, 1, bias=bias),
                                        nn.LeakyReLU(0.2, True),
                                         nn.Conv2d(f, f*2, 4, 2, 1, bias=bias))
        
        k = 0
        res_branch = []

        for i in range(4):
            # k = k + f*2
            res_branch.append(ResBlock_dense(f*2, f * 2))
            # res_branch.append(ResBlock_conv_dense(f*2, f * 2))

        self.res_branch = nn.Sequential(*res_branch)
        # ---------------------------------upsampling stage---------------------------------
        # ----------------------------------------------------------------------------------
        self.cond_stage2 = nn.Sequential(nn.Conv2d(f, f, 4, 2, 1, bias=bias),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Conv2d(f, f*2, 3, 1, 1, bias=bias))
        self.HR_branch = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                       nn.Conv2d(f * 2, f * 2, 3, 1, 1, bias=bias))
        res_branch1 = []
        for i in range(1):
            res_branch1.append(ResBlock_dense(f*2, f * 2))
            # res_branch1.append(ResBlock_conv_dense(f*2, f * 2))
        self.res_branch1 = nn.Sequential(*res_branch1)

        # ---------------------------------upsampling stage---------------------------------
        # ----------------------------------------------------------------------------------
        self.cond_stage3 = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1, bias=bias),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Conv2d(f, f*2, 3, 1, 1, bias=bias))
        
        self.HR_branch2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                        nn.Conv2d(f * 2, f * 2, 3, 1, 1, bias=bias))

        self.conv1 = nn.Conv2d(f * 4, f * 2, 3, 1, 1, bias=bias)
        
        res_branch2 = []
        for i in range(1):
            res_branch2.append(ResBlock_dense(f * 2, f * 2))
            # res_branch2.append(ResBlock_conv_dense(f * 2, f * 2))
        self.res_branch2 = nn.Sequential(*res_branch2)
        
        
        self.rec = nn.Conv2d(f * 2, inchannels, 3, 1, 1, bias=bias)
    def forward(self, ms_in, pan_in, grads, up_ms):
        # conditions
        cond = self.CondNet(grads)
        cond1 = self.cond_stage1(cond)
        cond2 = self.cond_stage2(cond)
        cond3 = self.cond_stage3(cond)
        # encoder
        ms_encoder = self.encoder21(up_ms)
        pan_encoder = self.encoder22(pan_in)
        ms_up1 = self.encoder31(ms_encoder)
        pan_up1 = self.encoder32(pan_encoder)
        ms_up2 = self.encoder41(ms_up1)
        pan_up2 = self.encoder42(pan_up1)
        concat = torch.cat((ms_up2, pan_up2), dim=1)
        fea = self.conv0(concat)
        feas = [fea]
        # fusion stage 1
        l1 = 6
        l2 = 4
        l3 = 2
        for i in range(l1):
            fea = self.res_branch((fea, feas[0], cond1))[0]
            # fea = self.res_branch((fea, feas[0], fea))[0]

        # upsampling stage 2
        fea = self.HR_branch(fea)
        feas.append(fea)
        
        for i in range(l2):
            fea = self.res_branch1((fea, feas[1], cond2))[0]
            # fea = self.res_branch1((fea, feas[1], fea))[0]
        fea = self.HR_branch2(fea)
        fea = torch.cat((ms_encoder, pan_encoder, fea), dim=1)
        fea = self.conv1(fea)
        
        feas.append(fea)
        # upsampling stage 3
        for i in range(l3):
            fea = self.res_branch2((fea, feas[2], cond3))[0]
            # fea = self.res_branch2((fea, feas[2], fea))[0]
        
        # rec
        fea = self.rec(fea)
        return torch.clamp(fea + up_ms,0,1)

class NETWORK:
    def __init__(self, opt, device):
        self.gnet = CreateGenerator(inchannels=opt.in_channels, grads=opt.grads)
        self.gnet.to(device=device)

        if torch.cuda.device_count() > 1:
            self.gnet = nn.DataParallel(self.gnet)
            # self.gnet = nn.parallel.DistributedDataParallel(self.gnet)
            # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
            # torch.distributed.init_process_group(backend="nccl")
            # self.gnet = nn.parallel.DistributedDataParallel(self.gnet)

            self.gnet.load_state_dict(
                torch.load(opt.load_dir0, map_location=device)
            )
        elif opt.load_dir0:
            self.gnet.load_state_dict(
                torch.load(opt.load_dir0, map_location=device)
            )
        else:
            init_weights(self.gnet, init_type='normal')

        self.normalL1Loss = nn.L1Loss()

        self.L1Loss = nn.L1Loss(reduction="sum")
        self.g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.gnet.parameters()),
                                     lr=opt.lr,
                                    #  betas=(0.9, 0.999),
                                     eps=1e-8,
                                     amsgrad=False)

        cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.device = device
        self.resduil7 = None
        self.opt = opt

        laplace = torch.tensor([[0, 1/4, 0],
                        [1/4, -1, 1/4],
                        [0, 1/4, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

        self.laplace = laplace.to(device=device)

        sobel = torch.tensor([[1/3., 1/3., 1/3.],
                        [0, 0, 0],
                        [-1/3., -1/3., -1/3.]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

        self.sobel = sobel.to(device=device)



        kernel_1d = cv2.getGaussianKernel(ksize=7, sigma=0, ktype=cv2.CV_64F)
        kernel_2d = kernel_1d * kernel_1d.T
        gaussian = torch.tensor(kernel_2d, dtype=torch.float, requires_grad=False).view(1, 1, 7, 7)
        
        gaussian = gaussian.repeat(opt.in_channels, 1, 1, 1)
        gaussian_filter = nn.Conv2d(in_channels=opt.in_channels, out_channels=opt.in_channels,kernel_size=7, groups=opt.in_channels, 
        bias=False, padding=3)

        gaussian_filter.weight.data = gaussian
        gaussian_filter.weight.requires_grad = False
    
        self.gaussian_filter = gaussian_filter.to(device=device)

    @staticmethod
    def conv_operator(img, kernel):
        y = F.conv2d(img, kernel, stride=1, padding=1,)
        return y

    def backward(self, inputs1, targets, inputs2, inputs3, inputs4, inputs2_full, inputs3_full, inputs4_full):
        self.g_optimizer.zero_grad()
        outputs = self.gnet(inputs1, inputs2, inputs3, inputs4)

        full_outputs = self.gnet(inputs4_full, inputs2_full, inputs3_full, inputs4_full)
        # basic
        ms_loss = self.normalL1Loss(targets, outputs)
        grad = self.conv_operator(inputs2_full, self.laplace)
        lap = self.conv_operator(torch.mean(full_outputs, dim=1, keepdim=True), self.laplace)
        pan_loss = self.normalL1Loss(lap, grad)

        self.g_loss = ms_loss + pan_loss*0.01
        self.g_loss.backward()
        self.g_optimizer.step()





