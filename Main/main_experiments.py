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
from train import trainModel_Exp
from generateDictionary import generateDictionary_Exp, convertStateDict



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
parser.add_argument('--layers', default=5, type=int,
					help='number of layers in model')
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
	layers = args.layers
	image_size = args.image_size
	exp_name = args.exp_name

	# Make sure the result directory exists.  If not create
	directory_logs = '../../EdgePixel_Results/Experiments/Logs'
	directory_results = '../../EdgePixel_Results/Experiments/ResultBlock'

	if not os.path.exists(directory_logs):
		os.makedirs(directory_logs)

	if not os.path.exists(directory_results):
		os.makedirs(directory_results)


	# Create name for result folders
	log_file = '../../EdgePixel_Results/Experiments/Logs/'+ exp_name + '.log'
	result_file = '../../EdgePixel_Results/Experiments/ResultBlock/resultBlock_' + exp_name + '.pth.tar'
	model_file = '../../EdgePixel_Results/Experiments/ResultBlock/modelBlock_' + exp_name + '.pth.tar'

	# Initizlize Logger
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	formatter = logging.Formatter('[%(asctime)s:%(name)s]:%(message)s')

	file_handler = logging.FileHandler(log_file)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	# Print experiment parameters to log
	logger.info('Training %s models with %i layers for %d epochs.' % (model_type, layers, n_epochs))
	logger.info('Number of models: %d' % (n_models))


	# Want to change this so that hyperparameter can only be loaded
	if os.path.isfile(hyper_path):
		print('Loading hyperparameter block.')
		hyperparameter = torch.load(hyper_path)
	else:
		print("=> no hyperparameter block found at '{}'".format(hyper_path))
		# hyperparameter = {}
		# hyperparameter["RecurrentGrid"] = {}
		# hyperparameter["RecurrentGrid"][25] = {"Learning": 1e-3, "Batch": 32, "Weight_Decay": 0}
		# hyperparameter["RecurrentGrid"][30] = {"Learning": 1e-3, "Batch": 32, "Weight_Decay": 0}
		# hyperparameter["RecurrentMasked5"] = {}
		# hyperparameter["RecurrentMasked5"][5] = {"Learning": 1e-4, "Batch": 32, "Weight_Decay": 1e-3}
		# hyperparameter["Recurrent"] = {}
		# hyperparameter["Recurrent"][5] = {"Learning": 1e-4, "Batch": 32, "Weight_Decay": 1e-3}


	# Set up experiment block
	num_nodes = image_size**2
	loss_fn = nn.MSELoss()
	dtype = torch.FloatTensor
	if args.use_gpu:
		print('GPU is used.')
		dtype = torch.cuda.FloatTensor
	

	if ((load_experiment) and os.path.isfile(load_experiment) and os.path.isfile(load_result)):
		modelBlock = torch.load(load_experiment)
		resultBlock = torch.load(load_result)
	else:
		print("=> Generating new result block")
		modelBlock, resultBlock = generateDictionary_Exp(n_models, model_type, layers, num_nodes, num_nodes,
			image_size, loss_fn, dtype, hyperparameter)



	# Figure out how many epochs are left to train
	epochs_remaining = n_epochs - modelBlock["Meta"]["Epochs_Trained"]

	trainModel_Exp(modelBlock, resultBlock, epochs_remaining, log_file, result_file, model_file)

	torch.save(resultBlock, result_file)

	modelBlock_State = convertStateDict(modelBlock)
	torch.save(modelBlock_State, model_file)


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)





