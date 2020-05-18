import configparser


# Method to read config file taken from https://stackoverflow.com/a/29925487/10062180
# CAN ONLY BE USED FOR READ ACCESS OF INI FILES
class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read(config_file_path):
    config = configparser.ConfigParser()
    #config = configparser.ConfigParser(dict_type=AttrDict)
    config.read(config_file_path)
    return config
