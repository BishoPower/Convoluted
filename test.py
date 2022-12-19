import numpy as np
from convoluted.nn.functions import swish, Rearranger
in_channels = 3
scale = 2 / np.sqrt(in_channels * 3 * 3)

weights = np.random.normal(scale=scale, size=(in_channels, in_channels, *(3, 3)))
bias = np.zeros(shape=(in_channels, 1))    

IMG = "grid-0006.png"
from PIL import Image
img = np.array(Image.open(IMG)).reshape((1, 512, 768))
print(img.shape)

