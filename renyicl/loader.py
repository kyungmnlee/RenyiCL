from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
import numpy as np

def scalemix(view1, view2):
    def random_bbox(lam, H, W):
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    _, h, w = view1.shape
    lam = np.random.uniform(low=0.0, high=1.0)
    bbx1, bby1, bbx2, bby2 = random_bbox(lam, h, w)
    view1[:, bbx1:bbx2, bby1:bby2] = view2[:, bbx1:bbx2, bby1:bby2]
    return view1

class MultiCropsTransform:
    def __init__(self, src_view, tar_view, loc_view, n_crops=0, enable_scalemix=False):
        self.src_view = src_view
        self.tar_view = tar_view
        self.loc_view = loc_view
        self.n_crops = n_crops
        self.enable_scalemix = enable_scalemix

    def __call__(self, x):
        crops = []
        if self.enable_scalemix:
            im1 = scalemix(self.src_view(x), self.src_view(x))
        else:
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