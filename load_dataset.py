import chainer
import cv2
from config import CONFIG
import numpy as np
import ipdb
import random


class load_dataset(chainer.dataset.DatasetMixin):
    def __init__(self, img_paths, augment, img_size):
        self.img_paths = img_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        img_sequence = []
        if CONFIG.max_img_seq == "max":
            paths = self.img_paths[i]
        elif isinstance(CONFIG.max_img_seq, int):
            paths = random.sample(self.img_paths[i], CONFIG.max_img_seq)
            paths.sort()
        else:
            print("dataset sampling error.")
            exit()

        for i in paths:
            img_sequence.append(self.img_open(i))

        img_sequence = self.img_augment(img_sequence)

        img_sequence = [((i/255) - 0.5)/0.5 for i in img_sequence]
        # img_sequence = [vgg.prepare(i, size=None) for i in img_sequence]
        return img_sequence

    def img_open(self, path):
        image = cv2.imread(path)
        w, h = self.img_size
        image = cv2.resize(image, (w, h))
        return image

    def img_augment(self, img_sequence):
        # aug_flip, aug_rotate, aug_shrink, #aug_color, #aug_hsv, #aug_blur, #aug_gray

        if CONFIG.random_flip is True:
            param = np.random.rand() > 0.5
            if param is True:
                tmp = [cv2.flip(i, 1) for i in img_sequence]
                img_sequence = tmp

        if (CONFIG.contrast is True) and (CONFIG.brightness is True):
            param_contrast = np.random.uniform(
                CONFIG.contrast_min, CONFIG.contrast_max)
            param_brightness = np.random.uniform(
                -CONFIG.brightness_max_delta, CONFIG.brightness_max_delta)
            tmp = [i*param_contrast + param_brightness for i in img_sequence]
            img_sequence = tmp
        img_sequence = [i.transpose(2, 0, 1) for i in img_sequence]

        return img_sequence
