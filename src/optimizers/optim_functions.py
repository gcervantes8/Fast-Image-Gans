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
def supported_losses():
    return _optimizers().keys()