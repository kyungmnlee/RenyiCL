from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
import numpy as np

class MultiCropsTransform:
    def __init__(self, src_view, tar_view, loc_view, n_crops=0):
        self.src_view = src_view
        self.tar_view = tar_view
        self.loc_view = loc_view
        self.n_crops = n_crops

    def __call__(self, x):
        crops = []
        im1 = self.src_view(x)
        crops.append(im1)
        im2 = self.tar_view(x)
        crops.append(im2)
        if self.n_crops > 0 :
            for v in range(self.n_crops): 
                im = self.loc_view(x)
                crops.append(im)
        return crops

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)