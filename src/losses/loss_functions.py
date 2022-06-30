import torch.nn as nn
from src.losses.hinge_loss import HingeLoss


def _losses():
    return {
        'mse': (nn.MSELoss(), 0, 1),
        'bce': (nn.BCELoss(), 0, 1),
        'hinge': (HingeLoss(), -1, 1)
    }


# Returns a set with the name of supported loss functions
def supported_losses():
    return _losses().keys()


# Given a loss function returns a 3-tuple
# The loss function, the fake label, and the real label
# Returns 3-tuple of None if the loss function is not supported
def supported_loss_functions(loss_name: str):
    loss_functions = _losses()
    if loss_name in loss_functions:
        return loss_functions[loss_name.lower()]
    return None
