# Author: Brett Larsen
# Code for generating samples of edge-connected pixels along with image labels
#	generateSamplesFull.py returns image and label of edge connected pixels
#	Use generateSamplesPropagate.py to generate images in the process of propagating labels


import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import tensorflow as tf
from weightMask import generateSquareWeightMask, generateGridWeightMask

N =15

weightMask = generateGridWeightMask(N)

pixelMask = np.reshape(weightMask[100, :], (N, N))


plt.imshow(pixelMask, cmap='RdBu',  interpolation='none')
#savefig('weightMask.pdf', bbox_inches = 'tight',
# 		pad_inches = 0)
plt.show()
