import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import math
import cv2
import numpy as np
import scipy.stats as st
import time
import random


def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel / kernel.max()
    return kernel
# create a vignetting mask
g_mask = gkern(560, 3)
g_mask = np.dstack((g_mask, g_mask, g_mask))
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    paddingl = (kernel_size - 1) // 2
    paddingr = kernel_size - 1 - paddingl
    pad = torch.nn.ReflectionPad2d((paddingl, paddingr, paddingl, paddingr))
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return nn.Sequential(pad, gaussian_filter)


class SynData:
    def __init__(self, device):
        self.g_mask = torch.tensor(g_mask.transpose(2, 0, 1)).to(device)
        self.device = device

    def __call__(self, t: torch.Tensor, r: torch.Tensor, k_sz):
        device = self.device
        t = t.pow(2.2)
        r = r.pow(2.2)
        t=t.type(torch.float32)
        r=r.type(torch.float32)

        sigma = k_sz[np.random.randint(0, len(k_sz))]
        att = 1.08 + np.random.random() / 10.0
        alpha2 = 1 - np.random.random() / 5.0
        sz = int(2 * np.ceil(2 * sigma) + 1)
        g_kernel = get_gaussian_kernel(sz, sigma)
        g_kernel = g_kernel.to(device)
        r_blur = g_kernel(r).float()
        blend = r_blur + t

        maski = (blend > 1).float()
        mean_i = torch.clamp(torch.sum(blend * maski, dim=(2, 3)) / (torch.sum(maski, dim=(2, 3)) + 1e-6),
                             min=1).unsqueeze_(-1).unsqueeze_(-1)
        r_blur = r_blur - (mean_i - 1) * att
        r_blur = r_blur.clamp(min=0, max=1)

        h, w = r_blur.shape[2:4]
        neww = np.random.randint(0, 560 - w - 10)
        newh = np.random.randint(0, 560 - h - 10)
        alpha1 = self.g_mask[:, newh:newh + h, neww:neww + w].unsqueeze_(0)

        r_blur_mask = r_blur * alpha1
        blend = r_blur_mask + t * alpha2

        t = t.pow(1 / 2.2)
        r_blur_mask = r_blur_mask.pow(1 / 2.2)
        blend = blend.pow(1 / 2.2)
        blend = blend.clamp(min=0, max=1)

        return t, r_blur_mask, blend.float(), alpha2

device='cuda'
syn=SynData(device)
k_sz = np.linspace(2, 5, 80)
def set_input(input):
    Is=[]
    with torch.no_grad():
        T = input['T'].to(device).float()
        R = input['R'].to(device).float()
        for i in range(5):
            time.sleep(0.1*np.random.randint(0,10))
            _, R, I, alpha = syn(T, R,k_sz)  # Synthesize data
            Is.append(I)


        return Is