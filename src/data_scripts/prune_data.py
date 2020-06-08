# -*- coding: utf-8 -*-
"""
Created on May 25, 2020

@author: Gerardo Cervantes

Purpose: This script is for gathering data from using an existing Keras model.
Takes images from screenshots in a single directory and creating a main directory, and putting the
image in the correct directory
"""


import numpy as np
import glob
import shutil
import os
from PIL import Image
from keras.preprocessing.image import img_to_array
import pathlib

def open_image(path):
    pil_img = Image.open(path).convert('RGB')
    copy_image = pil_img.copy()
    pil_img.close()
    return copy_image


# Crops and resize pil images from images of the whole game to images of the
# star number of the game, resize accordingly
# pre-process should be True if you want varying x and y coordinates and sizes (Done to make model robust)
def crop_and_resize_image(pil_img):

    pil_img = pil_img.resize((452, 345), Image.ANTIALIAS)  # Width,height
    img_width = pil_img.size[0]

    x, y = 380, 1
    size = img_width - 2 - x
    w, h = size + x, round(size / 1.675) + y

    pil_img = pil_img.crop((x, y, w , h ))
    pil_img = pil_img.resize((67, 40), Image.ANTIALIAS)  # Width,height

    return pil_img


if __name__ == "__main__":

    image_directory = r'D:\mario_data\pruned_mario-cam'
    pruned_img_dir = r'D:\mario_data\pruned_pruned_mario-cam'
    model_file_path = 'sm64Model7.hdf5'
    if not os.path.isdir(image_directory):
        raise OSError('Invalid image dir')

    if not os.path.isdir(pruned_img_dir):
        raise OSError('Invalid prune dir')

    exts = ['.png', '.jpg']
    img_paths = []
    for ext in exts:
        recursive_path = image_directory + '/**/*' + ext
        ext_imgs = glob.glob(recursive_path, recursive=True)
        img_paths = img_paths + ext_imgs

    print('Number of images: ' + str(len(img_paths)))

    from keras import backend as K
    K.clear_session()
    from keras.models import load_model
    model = load_model(model_file_path)

    for img_path in img_paths:
        image = open_image(img_path)
        resized_img = crop_and_resize_image(image)
        np_img = np.array(img_to_array(resized_img)).astype(np.float32)/255
        nn_output = model.predict(np.array([np_img]))
        is_bad_img = np.argmax(nn_output) == 121 or np.argmax(nn_output) == 122 or np.max(nn_output) < 0.5
        if is_bad_img:
            shutil.move(img_path, pruned_img_dir)