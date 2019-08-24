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
import logging, sys

# Import functions from submodules
from hyperOpt import hyperOpt
from generateDictionary import convertStateDict_Hyp

# Coding ToDo List for module
#		* Make final check more robust in case all losses > 100




##################################################################
# Parameters:
# 		* resume: path to hyperparameter block
#		* n_models: number of models with random parameters to generate
#		* n_epochs: number of epochs in between hyperband pruning
#				(total epochs for hyperband is 5*n_epochs)
#		* use_gpu: True/False for training on GPU
#		* models: model types to perform hyperparamer optimization over
#		* layers: layer numbers to perform hyperparameter optimization over
# Outputs
#		* hyperparameter: Nested dictionary of hyperparameters
#				(saved to disk)
##################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to hyperparameter block')
parser.add_argument('--exp_name', default='unnamed', type=str, metavar='PATH',
					help='name of experiment being run')
parser.add_argument('--n_models', default=100, type=int,
					help='number of models with random parameters to generate')
parser.add_argument('--n_epochs', default=100, type=int,
					help='number of epochs in between hyperband pruning')
parser.add_argument('--hyp_epochs', default=4, type=int,
					help='Total number of pruning rounds')
parser.add_argument('--use_gpu', action='store_true')
# parser.add_argument('--models', nargs='+', type=str,
# 					help='model types to perform hyperparamer optimization over')
parser.add_argument('--model', default='Recurrent', type=str,
					help='model to perform hyperparameter optimization over')
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
	hyp_epochs = args.hyp_epochs
	load = args.resume
	model_type = args.model
	layer_list = args.layers
	image_size = args.image_size
	exp_name = args.exp_name

	# Make sure the result directory exists.  If not create
	directory_logs = '../../RecurrentComputation_Results/Hyperoptimization/Performance'
	directory_parameters = '../../RecurrentComputation_Results/Hyperoptimization/Parameters'

	if not os.path.exists(directory_logs):
		os.makedirs(directory_logs)

	if not os.path.exists(directory_parameters):
		os.makedirs(directory_parameters)

	# Create name for result folders
	hyperopt_file = '../../RecurrentComputation_Results/Hyperoptimization/Parameters/hyperparameter_' + exp_name + '.pth.tar'
	model_file = '../../RecurrentComputation_Results/Hyperoptimization/Parameters/model_' + exp_name + '.pth.tar'
	log_file = '../../RecurrentComputation_Results/Hyperoptimization/Performance/'+ exp_name + '.log'
	result_file = '../../RecurrentComputation_Results/Hyperoptimization/Performance/resultBlock_' + exp_name

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
	logger.info('Training %s models for %d hyperband epochs of %d epochs each.' % (model_type, hyp_epochs, n_epochs))
	logger.info('Initial number of models: %d' % (n_models))

	
	# Setup experiment block
	num_nodes = image_size**2
	loss_fn = nn.MSELoss()
	dtype = torch.FloatTensor
	if args.use_gpu:
		print('GPU is used.')
		dtype = torch.cuda.FloatTensor

	# Make experiment block
	count = 0
	expBlock = {}
	for layer in layer_list:
		expBlock[count] = {"Network_Type": model_type, "Layers": layer, 
			"Input": num_nodes, "Hidden": num_nodes}
		count = count+1


	# Either Initialize the hyperparameter dictionary
	# or load it in if the resume flag is passed in
	# if args.resume:
	# 	if os.path.isfile(load):
	# 		hyperparameter = torch.load(load)
	# 	else:
	# 		print("=> no hyperparameter block found at '{}'".format(load))
	# else:
	hyperparameter = {}

	runHyperOpt(expBlock, hyperparameter, n_models, n_epochs, hyp_epochs, loss_fn, dtype, image_size, model_file, result_file, log_file)

	# Save out the updated hyperparameter block and the results form all the experiments
	torch.save(hyperparameter, hyperopt_file)








def runHyperOpt(expBlock, hyperparameter, n_models, n_epochs, hyp_epochs, loss_fn, dtype, image_size, model_file, result_file, log_file):
	##################################################################
	# Function RUN_HYPER_OPT
	# Takes in a nested dictionary of model/layer types
	# Outputs a nested dictionary of optimized hyperparameters
	# Parameters:
	# 		* expBlock: Nested dictionary of archtiectures
	#       * n_epochs: Number of epochs for which to train
	#       * N: size of image to generate
	# Outputs
	#		* hyperparameter: Nested dictionary of hyperparameters

	# Read in all the arguments that define the hyperOpt experiment
	##################################################################

	# Nested loop over experiments
	for key, val in expBlock.items():
		result_file_layer = result_file + '_' + str(expBlock[key]["Layers"]) + '.pth.tar'
		model_file_layer = model_file + '_' + str(expBlock[key]["Layers"]) + '.pth.tar'
		network = expBlock[key]["Network_Type"]
		layers = expBlock[key]["Layers"]
		lr, batch_size, weight_decay, acc, avg_loss, resultBlock = hyperOpt(n_models, n_epochs, hyp_epochs, network, 
			layers, expBlock[key]["Input"], expBlock[key]["Hidden"], image_size, dtype, log_file, result_file_layer, model_file_layer)
		if (not(network in hyperparameter)):
			hyperparameter[network] = {}
		if (not(layers in hyperparameter[network])):
			hyperparameter[network][layers] = {}
		hyperparameter[network][layers]["Learning"] = lr
		hyperparameter[network][layers]["Batch"] = batch_size
		hyperparameter[network][layers]["Weight_Decay"] = weight_decay
		hyperparameter[network][layers]["Acc"] = acc
		hyperparameter[network][layers]["Loss"] = avg_loss

		
		torch.save(resultBlock, result_file_layer)

	



if __name__ == '__main__':
	args = parser.parse_args()
	main(args)






