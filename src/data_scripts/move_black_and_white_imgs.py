# -*- coding: utf-8 -*-
"""
Created on April  17, 2021

Purpose: Move all black and white images in a directory to a different directory

"""
import PIL
from PIL import Image
import argparse
import os
import numpy as np
import shutil
from tqdm import tqdm
import multiprocessing


def move_if_bw_image_star(unpacked_arguments):
    move_if_bw_image(*unpacked_arguments)


def move_if_bw_image(image_path, move_dir):
    try:
        np_img = np.asarray(Image.open(image_path))

        # Average the RGB colors - Turns to grayscale
        np_img = np.mean(np_img, axis=2)
        black_thresh = 25
        white_thresh = 190

        pixel_count = np.size(np_img)
        black_pixel_count = np.sum(np_img < black_thresh)
        white_pixel_count = np.sum(np_img > white_thresh)

        black_ratio = float(black_pixel_count / pixel_count)
        white_ratio = float(white_pixel_count / pixel_count)
        # print("Black ratio: " + str(black_ratio) + "\tWhite ratio: " + str(white_ratio))
        if black_ratio > 0.75 or white_ratio > 0.75:
            shutil.move(image_path, move_dir)
            return True
    except IOError or PIL.UnidentifiedImageError or OSError:
        return False

    return False


# image_directory is a path to a directory that has images which you want to check if they are corrupt
def move_bw_images(image_directory, move_dir, num_threads):
    num_files = len(os.listdir(image_directory))
    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory)]
    with multiprocessing.Pool(num_threads) as pool:
        list(tqdm(pool.imap(move_if_bw_image_star, zip(image_paths, [move_dir] * num_files)), total=num_files))
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check for corrupt images in image folder.')
    parser.add_argument('image_dir', type=str,
                        help='a directory containing images to be processed')
    parser.add_argument('move_dir', type=str,
                        help='a directory which will get all of the black and white images')
    parser.add_argument('num_threads', type=int,
                        help='how many threads to run with')
    args = parser.parse_args()
    print('Moving black and white images ...')
    move_bw_images(args.image_dir, args.move_dir, args.num_threads)

