import torch.nn as nn

# get activation module from names
def get_activation(activation):

    if activation == 'relu':
        return nn.ReLU(inplace=True)
    else:
        raise NotImplementedError('Invalid activation name!') 
    # Wait for more kind of activations in the future!