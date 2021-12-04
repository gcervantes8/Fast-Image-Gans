# -*- coding: utf-8 -*-
"""
Created on November 28, 2021

@author: Gerardo Cervantes

Purpose: This script checks if a directory has any corrupt images.
Prints out images that are corrupt.
"""

import PIL
from PIL import Image
import argparse
import os


# image_directory is a path to a directory that has images which you want to check if they are corrupt
def check_for_corrupt(image_directory):
    corrupt_img_count = 0
    print('Looking for corrupt images ...')
    for filename in os.listdir(image_directory):
        image_path = os.path.join(image_directory, filename)
        try:
            Image.open(image_path)
        except IOError or PIL.UnidentifiedImageError or OSError:
            print('Problematic image: ' + image_path)
            corrupt_img_count += 1

    print(str(corrupt_img_count) + ' corrupt images found')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check for corrupt images in image folder.')
    parser.add_argument('image_dir', type=str,
                        help='a directory containing images which you want to look for corrupted images in')
    args = parser.parse_args()
    check_for_corrupt(args.image_dir)
