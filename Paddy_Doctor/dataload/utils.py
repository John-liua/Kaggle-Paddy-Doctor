from __future__ import division
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def rotate(img, angle, resample=False, expand=False, center=None):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.rotate(angle, resample, expand, center)
