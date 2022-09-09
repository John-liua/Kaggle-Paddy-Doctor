# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: preprocess.py
# Time: 6/27/19 1:32 PM
# Description: 
# -------------------------------------------------------------------------------

import math
import random
import numpy as np
import torch
from dataload.utils import *
import numbers
from torchvision.transforms import Resize, CenterCrop


class RandomErasing(object):

    def __init__(self, probability=0.5, sl=0.02, sh=1 / 3, r1=0.3, mean=None):
        if mean is None: mean = [0., 0., 0.]
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)

                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

                return img

        return img


class Cutout(object):

    def __init__(self, n_holes=256, length=4, probability=0.5):
        self.n_holes = n_holes
        self.length = length
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        holes = random.randint(64, self.n_holes)
        for n in range(holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            length = self.length

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomRotation(object):
    def __init__(self, degrees, probability=0.5, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.probability = probability

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        angle = self.get_params(self.degrees)
        return rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomSizedCrop:
    """Random crop the given PIL.Image to a random size
    of the original size and and a random aspect ratio
    of the original aspect ratio.
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR,
                 min_aspect=4 / 5, max_aspect=5 / 4,
                 min_area=0.25, max_area=1):
        self.size = size
        self.interpolation = interpolation
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area, self.max_area) * area
            aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize(self.size, self.interpolation)

        # Fallback
        scale = Resize(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


class RandomResizePadding:

    def __init__(self, size, interpolation=Image.BICUBIC, min_area=0.5, max_area=1, probability=0.75):
        self.size = size
        self.interpolation = interpolation
        self.min_area = min_area
        self.max_area = max_area
        self.probability = probability

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        #     return img.resize(self.size, self.interpolation)

        scale = random.uniform(self.min_area, self.max_area)
        img = img.resize((int(np.ceil(self.size[0] * scale)), int(np.ceil(self.size[1] * scale))), self.interpolation)
        img = np.array(img)

        start = int((self.size[0] - img.shape[0]) / 2)
        mask = np.zeros((self.size[0], self.size[1], 3), np.float32)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                mask[i + start, j + start, :] = img[i, j, :]

        mask = Image.fromarray(np.uint8(mask))

        return mask


class RandomResize:

    def __init__(self, size, interpolation=Image.BILINEAR, min_area=0.875, max_area=1):
        self.size = size
        self.interpolation = interpolation
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        scale = random.uniform(self.min_area, self.max_area)
        img = img.resize((int(np.ceil(self.size[0] * scale)), int(np.ceil(self.size[1] * scale))), self.interpolation)
        return img