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
from generateDictionary import generateDictionary_Hyperopt, convertStateDict_Hyp
from train import trainModel_Hyperopt, checkAccuracy

# Coding ToDo List for module
#		* Make final check more robust in case all losses > 100



def hyperOpt(n_model, n_epochs, hyp_epochs, model_type, layers, input_size, hidden_size, image_size, dtype, log_file, result_file, model_file):
	# Function HYPER_OPT
	# Implements the hyperband algorithm over hyperparameters: 
	#	learning rate and batch_size
	# Parameters:
	# 		* model_type: Specifies network architecture
	#       * layers, input_size, hidden_size: Network architecture paramters
	# Returns:
	#		* lr, batch_size: Learning rate and batch size of best performing model
	#		* acc, avg_loss: Accuracy and average loss of best performing model

	# Make dictionary of models with varying values of hyperparameters to optimize

	
	# Setup logger
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	formatter = logging.Formatter('[%(asctime)s:%(name)s]:%(message)s')

	if not len(logger.handlers):
		file_handler = logging.FileHandler(log_file)
		file_handler.setFormatter(formatter)

		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)

		logger.addHandler(file_handler)
		logger.addHandler(stream_handler)


	# Run hyperband epoch		
	loss_fn = nn.MSELoss()

	modelBlock, resultBlock = generateDictionary_Hyperopt(n_model, model_type, layers, 
		input_size, hidden_size, image_size, loss_fn, dtype)

	torch.save(resultBlock, result_file)
	modelBlock_State = convertStateDict_Hyp(modelBlock)
	torch.save(modelBlock_State, model_file)

	for h_epoch in range(hyp_epochs):
		trainModel_Hyperopt(modelBlock, n_epochs, log_file)
		pruneModel(modelBlock, resultBlock)
		torch.save(resultBlock, result_file)
		modelBlock_State = convertStateDict_Hyp(modelBlock)
		torch.save(modelBlock_State, model_file)

	trainModel_Hyperopt(modelBlock, n_epochs, log_file)

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
	modelBlock_State = convertStateDict_Hyp(modelBlock)
	torch.save(modelBlock_State, model_file)
	
	return lr, batch_size, weight_decay, acc, avg_loss, resultBlock


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






