import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random

import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from skimage import metrics
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


import torch
import numpy as np
import os
from PIL import Image
import random
import torchvision.transforms.functional as Ft
from torchvision import transforms, utils, models
import  torch.nn as nn

    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return (-1)*ssim_map.mean()
    else:
        return (-1)*ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
 

def random_crop(data, label, psize):
    angRes, angRes, c, h, w = data.shape
    
    x = random.randrange(0, h - psize, 16)
    # print(x)
    y = random.randrange(0, w - psize, 16)

    data = data[:, :, :, x:x+psize, y:y+psize]
    label = label[:, :, :, x:x+psize, y:y+psize]

    return data, label
   


class DataSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(DataSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = os.listdir(dataset_dir)
        self.item_num = len(self.file_list)
        

    def __getitem__(self, index):
        file_name = self.dataset_dir + '/' + self.file_list[index]
        
        with h5py.File(file_name, 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))  #(c,w,h,v,u)
            
            data = np.transpose(data, (4, 3, 0, 2, 1))
            label = np.transpose(label, (4, 3, 0, 2, 1))
            
            data = torch.from_numpy(data)  # (u,v,c,h,w)=(7,7,64,64,3)
            label = torch.from_numpy(label)
            
            ##### If 7x7 #####
            # data = data[1:6, 1:6, :, :, :]
            # label = label[1:6, 1:6, :, :,]
            
        return data, label 

    def __len__(self):
        return self.item_num 
    
def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out

def LFdivide(lf, patch_size, stride):
    U, V, C, H, W = lf.shape
    data = rearrange(lf, 'u v c h w -> (u v) c h w')

    bdr = (patch_size - stride) // 2
    numU = (H + bdr * 2 - 1) // stride
    numV = (W + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(u v) (c h w) (n1 n2) -> n1 n2 u v c h w',
                      n1=numU, n2=numV, u=U, v=V, h=patch_size, w=patch_size)

    return subLF

def LFintegrate(subLFs, patch_size, stride):
    n1, n2, u, v, c, h, w = subLFs.shape
    bdr = (patch_size - stride) // 2
    outLF = subLFs[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 u v c h w ->u v c (n1 h) (n2 w)')
    return outLF

def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np)

def cal_ssim(img1, img2):
    ssim = 0
    for i in range(img1.shape[0]):
        ssim += metrics.structural_similarity(img1[i,...], img2[i,...], gaussian_weights=True, data_range=1.0)

    return ssim/img1.shape[0]

def cal_metrics(label, out):

    U, V, C, H, W = label.size()
    label = label.data.cpu().numpy().clip(0, 1)
    out = out.data.cpu().numpy().clip(0, 1)

    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')
    for u in range(U):
        for v in range(V):
            PSNR[u, v] = metrics.peak_signal_noise_ratio(label[u, v, :, :, :], out[u, v, :, :, :])
            SSIM[u, v] = cal_ssim(label[u, v, :, :, :], out[u, v, :, :, :])

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean


def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st
