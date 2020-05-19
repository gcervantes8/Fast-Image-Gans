# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:12:31 2020

@author: Gerardo Cervantes

Purpose: Additional functions for dealing with directories and paths
"""

import os
import glob
import string
import random
import numpy as np


# This function throws exception if the directory doesn't exist, and gives the given error message
def is_valid_dir(dir_path, error_msg):
    if not os.path.isdir(dir_path):
        raise OSError(error_msg)


# Creates a directory with given name run_id in the output_dir
# Returns 2-tuple (path, run_id)
def create_run_dir(output_dir):
    run_id = _id_generator()
    is_valid_dir(output_dir, output_dir + '\nIs not a valid directory')
    try:
        run_dir_path = os.path.join(output_dir, run_id)
        
        os.mkdir(run_dir_path)
        print('Directory ', run_dir_path,  ' Created ')
        return run_dir_path, run_id
    except FileExistsError:
        raise OSError('Could create new directory with identifier: ' + run_id +
                      '\nIt\'s possible one exists already in the ' + output_dir)


# Creates random combination of ascii and numbers of given size
# taken from https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def _id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# Returns configuration file (.ini extension) in a given directory
def find_config_file(dir_path):
    config_files_in_dir = glob.glob(dir_path + '/*.ini', recursive=False)
    return config_files_in_dir[0]


# Finds latest (most trained) pytorch generator in run directory
# Assumes file name starts with generator, and the epoch number is after an underscore and before a period
def find_latest_generator_model(run_dir):
    model_files = glob.glob(run_dir + '/*.pt', recursive=False)
    gen_model_files = [model_file for model_file in model_files if 'generator_' in model_file]

    # Removes extension of file paths
    gen_model_names = [gen_model_file.split('.')[0] for gen_model_file in gen_model_files]

    # Gets last thing after _
    gen_epoch_nums = [gen_model_name.split('_')[-1] for gen_model_name in gen_model_names]

    model_epoch_nums = [int(num) if num.isdigit() else -1 for num in gen_epoch_nums]
    latest_model_index = np.argmax(np.array(model_epoch_nums))
    return gen_model_files[latest_model_index]






