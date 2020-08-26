# -*- coding: utf-8 -*-

"""
データをnumpy配列で保存
"""

import glob
import numpy as np
from scipy import ndimage


def main():
    images = []
    for file in glob.glob('images/*.png'):
        images.append(ndimage.imread(file, mode='RGB'))
    np.save('mini-cifar10.npy', np.array(images, dtype=np.uint8))


if __name__ == '__main__':
    main()
