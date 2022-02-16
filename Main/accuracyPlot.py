import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable 
import copy
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import math
import time
from scipy.io import savemat
import collections
import shutil

# Import functions from submodules
from train import perPixelAccuracy
import networkFiles as NF
from samples import generateSamples
#from hyperOpt import hyperOpt
#from generateDictionary import generateDictionary_Exp, generateDictionary_Hyperopt, convertStateDict_Hyp


# Coding ToDo List for module
#		* Make final check more robust in case all losses > 100




##################################################################
# Parameters:
# 		* resume: path to hyperparameter block
#		* n_models: number of models with random parameters to generate
#		* n_epochs: number of epochs in between hyperband pruning
#				(total epochs for hyperband is 12*n_epochs)
#		* use_gpu: True/False for training on GPU
#		* models: model types to perform hyperparamer optimization over
#		* layers: layer numbers to perform hyperparameter optimization over
# Outputs
#		* hyperparameter: Nested dictionary of hyperparameters
#				(saved to disk)
##################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to experiment block to continue')
parser.add_argument('--resume_result', default='', type=str, metavar='PATH',
					help='path to result block to continue')
parser.add_argument('--hyper', default='', type=str, metavar='PATH',
					help='path to hyperparameter block')
parser.add_argument('--exp_name', default='unnamed', type=str, metavar='PATH',
					help='name of experiment being run')
parser.add_argument('--n_models', default=100, type=int,
					help='number of models with random parameters to generate')
parser.add_argument('--n_epochs', default=100, type=int,
					help='number of total epochs we want the model trained')
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--model', default='Recurrent', type=str,
					help='model types to perform hyperparamer optimization over')
parser.add_argument('--layers', nargs='+', type=int,
					help='layer numbers to perform hyperparameter optimization over')
parser.add_argument('--image_size', default=15, type=int,
					help='number of epochs in between hyperband pruning')


def main(args):
	##################################################################
	# Top level code for running hyperoptimization
	# User specifies model type and layer number
	# Code then finds optimal hyperparameters for all
	# 		combinations of models/layers
	##################################################################

	# Load in arguments
	n_models = args.n_models
	n_epochs = args.n_epochs
	load_experiment = args.resume
	load_result = args.resume_result
	hyper_path = args.hyper
	model_type = args.model
	layer_list = args.layers
	image_size = args.image_size
	exp_name = args.exp_name



	# Set up experiment block
	num_nodes = image_size**2
	loss_fn = nn.MSELoss()
	dtype = torch.FloatTensor
	if args.use_gpu:
		print('GPU is used.')
		dtype = torch.cuda.FloatTensor

	n = 25
	N = 12
	N2 = N*N
	distribution = np.ones(n)/n

	layer = [5]

	loss_array = []
	accAll_array = []
	accPath_array = []
	accDistract_array = []
	batch = 100

	print("Generating Samples")
	testDict = generateSamples(N, distribution, 1000, test=True)
	for i in layer:
		# Generate model with the appropriate layer
		print(i)
		model = NF.RecurrentScaledGridFixed(N2, N2, N2, i, N)
		accAll = perPixelAccuracy(model, loss_fn, dtype, batch, testDict)
		print(accAll.size())
		accAll = np.asarray(accAll)
		accAll = np.reshape(accAll, (N,N))

		# loss_array.append(avg_loss)
		# accAll_array.append(accAll)
		# accPath_array.append(accPath)
		# accDistract_array.append(accDistract)
		np.set_printoptions(precision=2)
		print(np.asarray(accAll))

	# layer_array = np.asarray(layer)
	# loss_array = np.asarray(loss_array)
	# accAll_array = np.asarray(accAll_array)
	# accPath_array = np.asarray(accPath_array)
	# accDistract_array = np.asarray(accDistract_array)

	# resultBlock = {"Layer": layer_array, "Loss": loss_array, "All": accAll_array, "Path": accPath_array, "Distract": accDistract_array}
	# print(resultBlock["All"])
	# print(resultBlock["Path"])
	# print(resultBlock["Distract"])

	# torch.save(resultBlock, "Grid.pth.tar")



if __name__ == '__main__':
	args = parser.parse_args()
	main(args)





