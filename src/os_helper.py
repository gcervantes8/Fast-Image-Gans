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
from enum import Enum


class ModelType(Enum):
    GENERATOR = 'gen'
    DISCRIMINATOR = 'discrim'
    EMA = 'ema'


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
        return run_dir_path, run_id
    except FileExistsError:
        raise OSError('Could not create new directory with identifier: ' + run_id +
                      '\nIt\'s possible one exists already in the following directory: ' + output_dir)


# Creates a directory with name of dir_nam in the given path, returns the path to the created directory
def create_dir(path, dir_name):
    dir_path = os.path.join(path, dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


# Creates random combination of ascii and numbers of given size
# taken from https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def _id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# Returns configuration file (.ini extension) in a given directory
def find_config_file(dir_path):
    config_files_in_dir = glob.glob(dir_path + '/*.ini', recursive=False)
    return config_files_in_dir[0]


# model_type should be a value from the enum of os_helper.ModelType
# If step_num is None, then it will load the latest model in the directory
# Assumes file name starts with generator, and the epoch number is after an underscore and before a period
def get_step_model(model_dir, model_type, step_num=None):
    model_files = glob.glob(model_dir + '/*.pt', recursive=False)

    model_type_files = [model_file for model_file in model_files if model_type.value + '_' in model_file]
    if not model_type_files:
        raise ValueError('No files found with model type: ' + model_type.value)

    # Removes extension of file paths
    model_names = [os.path.splitext(model_file)[0] for model_file in model_type_files]

    # Gets step num located after the last underscore
    str_step_numbers = [model_name.split('_')[-1] for model_name in model_names]

    int_step_numbers = [int(num) if num.isdigit() else -1 for num in str_step_numbers]

    # If no step number is specified, get the model with the highest step count
    if not step_num:
        step_num = max(int_step_numbers)

    model_index = int_step_numbers.index(step_num)

    return model_type_files[model_index], step_num
