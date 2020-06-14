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
import multiprocessing
from keras.preprocessing.image import img_to_array

# Multi-process safe way to load model
def load_model(model_file_path):
    from keras import backend as K
    import tensorflow as tf
    # Load model
    from keras.models import load_model
    model = load_model(model_file_path)
    # For multi-threading
    session = K.get_session()
    model._make_predict_function()
    graph = tf.compat.v1.get_default_graph()
    graph.finalize()
    return model, session, graph
    

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

def paths_to_numpy(paths):
    np_images = []
    for path in paths:
        image = Image.open(path).convert('RGB')
        image = crop_and_resize_image(image)
        np_images.append(img_to_array(image))
        image.close()
    np_images = np.array(np_images).astype(np.float32) / 255
    return np_images


# Images with white or black screen or low prediction values will be moved from src location to dest location
def move_uncertain_images(src_paths, dst_path, model_file_path):
    print('Process starting')
    # Run in batches, to keep memory use low, reduce batch size if too much memory is being used
    batch_size = 256
    print('loading model')
    model, session, graph = load_model(model_file_path)
    print('loaded model')
    is_more_batches = True
    n_process_imgs = len(src_paths)
    while is_more_batches:
        if len(src_paths) <= batch_size:
            is_more_batches = False
            
        batch_paths = src_paths[:batch_size]
        src_paths = src_paths[batch_size:]        
        np_images = paths_to_numpy(batch_paths)
        
        with session.as_default():
            with graph.as_default():
                nn_outputs = model.predict(np_images)

        for i, img_pred in enumerate(nn_outputs):
            pred = np.argmax(img_pred)
            pred_prob = np.max(img_pred)
            is_bad_img = pred == 121 or pred == 122 or pred_prob < 0.5
            if is_bad_img:
                shutil.move(batch_paths[i], dst_path)
        percent_done = ((n_process_imgs-len(src_paths))/n_process_imgs) * 100
        print('Process: ' + str(percent_done) + '% done')
    print('Process finished')

if __name__ == "__main__":

    image_directory = r'lozplyr'
    pruned_img_dir = r'lozplyr_pruned'
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

    n_processes = 7.0
    process_batch_size = int((len(img_paths) / n_processes))
    print('Batch size for each process: ' + str(process_batch_size))
    is_finished = False
    processes = []
    while not is_finished:

        if len(img_paths) <= process_batch_size:
            # call with this thread
            move_uncertain_images(img_paths, pruned_img_dir, model_file_path)
            is_finished = True
        else:
            batch_img_paths = img_paths[:process_batch_size]
            process = multiprocessing.Process(target=move_uncertain_images, args=(batch_img_paths,pruned_img_dir,model_file_path,))
            process.start()
            processes.append(process)
            img_paths = img_paths[process_batch_size:]

    for process in processes:
        process.join()
    print('Program finished')
