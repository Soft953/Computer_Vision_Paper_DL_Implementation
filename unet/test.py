import numpy as np

import torch.nn.functional as F

from model import UNet


def test():
    model = UNet(n_classes=2)
    
    # fake data
    inputs = F.Tensor(np.random.normal(0, 1, size=(1, 3, 256, 256)))

    #test with fake data
    outputs = model(inputs)

    print(outputs.shape)

    assert outputs.shape == (1, 3, 256, 256)

if __name__ == "__main__":
    test()