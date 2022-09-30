# -*- coding: utf-8 -*-
"""
Created on Thu April 30 2020

@author: Gerardo Cervantes

Purpose: For reading the configuration file
"""

import configparser
import pathlib
import os


# Method to read config file taken from https://stackoverflow.com/a/29925487/10062180
# CAN ONLY BE USED FOR READ ACCESS OF INI FILES
class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_ini(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config


def append_defaults_configurations(project_config, default_config):
    config = {}
    config.update(default_config)
    for config_type in project_config:
        for key in project_config[config_type]:
            if config_type in config:
                config[config_type][key] = project_config[config_type][key]
            else:
                config[config_type] = {key: project_config[config_type][key]}
    return config


def read_with_defaults(config_file_path):
    config = read_ini(config_file_path=config_file_path)
    try:
        model_name = config['MODEL ARCHITECTURE']['model_type']
    except KeyError:
        raise ValueError('Model is not specified in the configuration file provided')
    project_dir = pathlib.Path().resolve()
    model_default_config_path = os.path.join(project_dir, 'src', 'models', model_name, 'defaults.ini')
    if not pathlib.Path(model_default_config_path).exists():
        raise ValueError('Model: ' + model_name + ' is missing a default config')
    default_config = read_ini(config_file_path=model_default_config_path)
    return append_defaults_configurations(config, default_config)
