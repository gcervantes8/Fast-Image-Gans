# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:12:31 2020

@author: Gerardo Cervantes

Purpose: Additional functions for dealing directories and paths
"""

import os
import string
import random


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

