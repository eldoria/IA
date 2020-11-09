import os
from PIL import Image

IMAGES_SRC = 'images_src/'
IMAGES_RESIZE = 'images_resize/'
SIZE = 256, 256

if __name__ == "__main__":
    for file in os.listdir(IMAGES_SRC):
        image_path = IMAGES_SRC + file
        img = Image.open(image_path)
        img = img.resize(SIZE)
        img = img.convert('L')
        img.save(IMAGES_RESIZE + file)
