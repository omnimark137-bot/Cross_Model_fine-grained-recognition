import random
import math


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
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


class RandomTranslateCrop(object):
    """Randomly translate (shift) the crop center then crop to target size.

    This augmentation helps simulate moderate misalignment between paired
    modalities (e.g., optical vs SAR) by applying independent random
    translations before converting to tensor.

    Args:
        size (int or tuple): target output size (w,h) or integer for square.
        max_translate (float): maximum fraction of image width/height to
            translate the crop center (e.g., 0.2 = up to +-20%% shift).
    """

    def __init__(self, size, max_translate=0.2):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.max_translate = float(max_translate)

    def __call__(self, img):
        # img is a PIL Image
        w, h = img.size
        tw, th = self.size

        # if target is larger than image, fallback to center crop
        if tw > w or th > h:
            left = max(0, (w - tw) // 2)
            upper = max(0, (h - th) // 2)
            return img.crop((left, upper, left + tw, upper + th))

        max_dx = int(self.max_translate * w)
        max_dy = int(self.max_translate * h)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)

        left = (w - tw) // 2 + dx
        upper = (h - th) // 2 + dy

        # ensure crop box within image
        left = max(0, min(left, w - tw))
        upper = max(0, min(upper, h - th))

        return img.crop((left, upper, left + tw, upper + th))

