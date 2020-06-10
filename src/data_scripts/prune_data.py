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
from threading import Thread
from keras import backend as K
import tensorflow as tf


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


# Images with white or black screen or low prediction values will be moved from src location to dest location
def move_uncertain_images(src_image_paths, dst_path, model, session, graph):
    resized_images = []
    for src_image_path in src_image_paths:
        image = open_image(src_image_path)
        image = crop_and_resize_image(image)
        resized_images.append(image)

    np_images = [img_to_array(resized_image) for resized_image in resized_images]
    np_images = np.array(np_images).astype(np.float32) / 255
    with session.as_default():
        with graph.as_default():
            nn_outputs = model.predict(np_images)
    # Can be optimized
    for i, img_prediction in enumerate(nn_outputs):
        pred = np.argmax(img_prediction)
        pred_prob = np.max(img_prediction)
        is_bad_img = pred == 121 or pred == 122 or pred_prob < 0.5
        if is_bad_img:
            shutil.move(src_image_paths[i], dst_path)


if __name__ == "__main__":

    image_directory = r'fazerlazer'
    pruned_img_dir = r'fazerlazer_pruned'
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

    # Load model
    K.clear_session()
    from keras.models import load_model
    nn_model = load_model(model_file_path)
    # For multi-threading
    session = K.get_session()
    nn_model._make_predict_function()
    graph = tf.get_default_graph()
    graph.finalize()
    batch_size = 256
    is_finished = False
    threads = []
    while not is_finished:

        if len(img_paths) <= batch_size:
            # call with this thread
            move_uncertain_images(img_paths, pruned_img_dir, nn_model, session, graph)
            is_finished = True
        batch_img_paths = img_paths[:batch_size]
        thread = Thread(target=move_uncertain_images, args=(batch_img_paths,pruned_img_dir,nn_model,session, graph,))
        thread.start()
        threads.append(thread)
        img_paths = img_paths[batch_size:]

    for thread in threads:
        thread.join()
