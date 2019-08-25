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
from generateDictionary import generateDictionary_Hyperopt, convertStateDict
from train import trainModel, checkAccuracy

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
parser.add_argument('--model', default='Recurrent', type=str,
					help='model to perform hyperparameter optimization over')
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
	hyp_epochs = args.hyp_epochs
	load = args.resume
	model_type = args.model
	layers = args.layers
	image_size = args.image_size
	exp_name = args.exp_name

	# Make sure the result directory exists.  If not create
	directory_logs = '../../EdgePixel_Results/Hyperoptimization/Performance'
	directory_parameters = '../../EdgePixel_Results/Hyperoptimization/Parameters'

	if not os.path.exists(directory_logs):
		os.makedirs(directory_logs)

	if not os.path.exists(directory_parameters):
		os.makedirs(directory_parameters)

	# Create name for result folders
	hyperopt_file = '../../EdgePixel_Results/Hyperoptimization/Parameters/hyperparameter_' + exp_name + '.pth.tar'
	model_file = '../../EdgePixel_Results/Hyperoptimization/Parameters/model_' + exp_name + '.pth.tar'
	log_file = '../../EdgePixel_Results/Hyperoptimization/Performance/'+ exp_name + '.log'
	result_file = '../../EdgePixel_Results/Hyperoptimization/Performance/resultBlock_' + exp_name + '.pth.tar'

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

	
	# Setup network parameters
	num_nodes = image_size**2
	input_size = num_nodes
	hidden_size = num_nodes
	loss_fn = nn.MSELoss()
	dtype = torch.FloatTensor
	if args.use_gpu:
		print('GPU is used.')
		dtype = torch.cuda.FloatTensor

	hyperparameter = {}



	# Run hyperband epoch		
	modelBlock, resultBlock = generateDictionary_Hyperopt(n_models, model_type, layers, 
		input_size, hidden_size, image_size, loss_fn, dtype)

	torch.save(resultBlock, result_file)
	modelBlock_State = convertStateDict_Hyp(modelBlock)
	torch.save(modelBlock_State, model_file)

	for h_epoch in range(hyp_epochs):
		trainModel(modelBlock, n_epochs, log_file)
		pruneModel(modelBlock, resultBlock)
		torch.save(resultBlock, result_file)
		modelBlock_State = convertStateDict_Hyp(modelBlock)
		torch.save(modelBlock_State, model_file)

	trainModel(modelBlock, n_epochs, log_file)

	epoch_total = modelBlock["Meta"]["Epochs_Trained"]
	resultBlock["Meta"]["Total_Epochs"] = epoch_total

	# Find the model id with best loss and return its parameters
	best_loss = 1000.0
	for key, val in modelBlock.items():
		if (key != "Meta"):
			resultBlock[key][epoch_total] = {"Loss": modelBlock[key]["Loss"], "Acc_All": modelBlock[key]["Acc_All"],
				"Acc_Path": modelBlock[key]["Acc_Path"], "Acc_Distract": modelBlock[key]["Acc_Distract"]}
			resultBlock[key]["Hyperparameter"]["Max_Epoch"] = epoch_total

			if ((modelBlock[key]["Loss"] < best_loss)):
				best_loss = modelBlock[key]["Loss"]
				best_key = key

	# This ensures that values are returned even if none of the keys have loss >= 1000.0
	# This should not happen so print an error to the log
	if (best_loss >= 1000.0):
		logger.warning('All models had loss greater than 1000.0')
		logger.warning('Returning parameters for first remaining model')
		keys = list(modelBlock.keys())
		keys.remove("Meta")
		best_key = next(iter(keys))

	
	lr = modelBlock[best_key]["Learning"]
	batch_size = modelBlock[best_key]["Batch"]
	weight_decay = modelBlock[best_key]["Weight_Decay"]
	acc = modelBlock[best_key]["Accuracy"]
	avg_loss = modelBlock[best_key]["Loss"]

	resultBlock["Meta"]["Learning"] = modelBlock[best_key]["Learning"]
	resultBlock["Meta"]["Batch"] = modelBlock[best_key]["Batch"]
	resultBlock["Meta"]["Weight_Decay"] = modelBlock[best_key]["Weight_Decay"]
	resultBlock["Meta"]["Acc_All"] = modelBlock[best_key]["Acc_All"]
	resultBlock["Meta"]["Acc_Path"] = modelBlock[best_key]["Acc_Path"]
	resultBlock["Meta"]["Acc_Distract"] = modelBlock[best_key]["Acc_Distract"]
	resultBlock["Meta"]["Loss"] = modelBlock[best_key]["Loss"]
	resultBlock["Meta"]["Best_Key"] = best_key

	torch.save(resultBlock, result_file)
	modelBlock_State = convertStateDict(modelBlock)
	torch.save(modelBlock_State, model_file)


	if (not(model_type in hyperparameter)):
		hyperparameter[model_type] = {}
	if (not(layers in hyperparameter[model_type])):
		hyperparameter[model_type][layers] = {}
	hyperparameter[model_type][layers]["Learning"] = lr
	hyperparameter[model_type][layers]["Batch"] = batch_size
	hyperparameter[model_type][layers]["Weight_Decay"] = weight_decay
	hyperparameter[model_type][layers]["Acc"] = acc
	hyperparameter[model_type][layers]["Loss"] = avg_loss

		
	torch.save(resultBlock, result_file)
	torch.save(hyperparameter, hyperopt_file)



def pruneModel(modelBlock, resultBlock):
	# Function PRUNE_MODELS
	# Takes a modelBlock and modelList and returns the top half of performers
	# Deletes remaining models form the modelBlock and modelList
	# Parameters:
	# 		* modelBlock: Nested dictionary of models
	#       * modelList: List with key values of reamining models

	# First need to obtain all the loss values
	epoch_total = modelBlock["Meta"]["Epochs_Trained"]
	resultBlock["Meta"]["Total_Epochs"] = epoch_total

	loss = []
	for key, val in modelBlock.items():
		if (key != "Meta"):
			loss.append(modelBlock[key]["Loss"])

			# Update all the results for the trained models
			resultBlock[key][epoch_total] = {"Loss": modelBlock[key]["Loss"], "Acc_All": modelBlock[key]["Acc_All"],
				"Acc_Path": modelBlock[key]["Acc_Path"], "Acc_Distract": modelBlock[key]["Acc_Distract"]}
			resultBlock[key]["Hyperparameter"]["Max_Epoch"] = epoch_total


	loss_array = np.asarray(loss)
	loss_median = np.median(loss)

	selectedKeys = list()
	
	# Delete models with loss values grater than the median
	for key, val in modelBlock.items():
		if (key != "Meta"):
			if ((modelBlock[key]["Loss"] > loss_median)):
				selectedKeys.append(key)

	for key in selectedKeys:
		if key in modelBlock:
			del modelBlock[key]



if __name__ == '__main__':
	args = parser.parse_args()
	main(args)






