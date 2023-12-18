import bitsandbytes as bnb
import torch.optim as optim

def _optimizers():
    return {
        'Adam': optim.Adam,
        'Adam8': bnb.optim.Adam8bit,
        'RMSprop': optim.RMSprop,
        'SGD': optim.SGD,
        'Lion': bnb.optim.Lion,
        'Lion8': bnb.optim.Lion8bit
    }

# Returns a set with the name of supported loss functions
def supported_optimizer_list():
    return _optimizers().keys()

# Given a loss function returns a 3-tuple
# The loss function, the fake label, and the real label
# Returns 3-tuple of None if the loss function is not supported
def optimizer_factory(optimizer_str: str):
    optimizer_functions = _optimizers()
    if optimizer_str in optimizer_functions:
        return optimizer_functions[optimizer_str]
    return None