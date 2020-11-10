import numpy as np
from PIL import Image
import os
import Pooling

IMAGES_RESIZE = 'images_resize/'
IMAGES_CONVOLVED = 'images_convolved/'
IMAGES_AFTER_POOLING = 'images_pooled/'


class Convolution:
    blurring = 1/256 * np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])

    detect_edges_vertical = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    detect_edges_horizontal = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    @staticmethod
    def apply_convolution(img_src, kernel):
        img_src_np = np.array(img_src, dtype=np.float)
        img_target_np = np.zeros_like(img_src_np)
        img_padded_np = np.pad(img_src, kernel.shape[0] // 2)

        for line in range(img_src.height):
            for col in range(img_src.width):
                patch = img_padded_np[line:line + kernel.shape[0], col:col + kernel.shape[1]]
                img_target_np[line, col] = np.sum(patch * kernel)
        img_target_np = Image.fromarray(img_target_np).convert('L')
        return img_target_np


if __name__ == "__main__":
    c = Convolution()
    for file in os.listdir(IMAGES_RESIZE):
        img = Image.open(IMAGES_RESIZE + file)
        img = c.apply_convolution(img, c.detect_edges_horizontal)
        img.save(IMAGES_CONVOLVED + file)
        img = Pooling.pooling(img, "max")
        img.save(IMAGES_AFTER_POOLING + file)
