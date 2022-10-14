import pytest
from src.train_gan import train
import logging


def test_dcgan_train():
    logging.basicConfig(level=logging.DEBUG)
    test_file_path = 'src/tests/dcgan_test_config.ini'
    train(test_file_path)


def test_biggan_train():
    logging.basicConfig(level=logging.DEBUG)
    test_file_path = 'src/tests/biggan_test_config.ini'
    train(test_file_path)


def test_deep_biggan_train():
    logging.basicConfig(level=logging.DEBUG)
    test_file_path = 'src/tests/deep_biggan_test_config.ini'
    train(test_file_path)
