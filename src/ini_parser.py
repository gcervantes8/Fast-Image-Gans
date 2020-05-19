# -*- coding: utf-8 -*-
"""
Created on Thu April 30 2020

@author: Gerardo Cervantes

Purpose: For reading the configuration file
"""


import configparser


# Method to read config file taken from https://stackoverflow.com/a/29925487/10062180
# CAN ONLY BE USED FOR READ ACCESS OF INI FILES
class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config
