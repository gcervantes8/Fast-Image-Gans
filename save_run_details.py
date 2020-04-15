# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:12:31 2020

@author: Jerry C
"""

import string
import os
import random

# This function throws exception if the directory doesn't exist, and gives the given error message
def is_valid_dir(dir_path, error_msg):
    if not os.path.isdir(dir_path):
        raise OSError(error_msg)
    
# Creates random combination of ascii and numbers, taken from https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def make_run_dir(output_dir, run_id):
    is_valid_dir(output_dir, output_dir + '\nIs not a valid directory')
    try:
        run_dir_path = os.path.join(output_dir, run_id)
        
        os.mkdir(run_dir_path)
        print("Directory " , run_dir_path ,  " Created ") 
        return run_dir_path
    except FileExistsError:
        raise OSError('Could create new directory with identifier: ' + run_id +
                      '\nIt\'s possible one exists already in the ' + output_dir)