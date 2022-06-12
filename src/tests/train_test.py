import pytest
from src.train_gan import train
import logging


def test_default_train():
    logging.basicConfig(level=logging.DEBUG)
    test_file_path = 'src/tests/test_config.ini'
    train(test_file_path)

