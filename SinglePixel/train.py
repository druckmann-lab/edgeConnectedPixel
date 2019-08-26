import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable 
import copy
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import math
import time
from samples import generateSamples
from scipy.io import savemat
import collections
import shutil
import networkFiles as NF
import logging, sys

from generateDictionary import convertStateDict


# Coding ToDo List for module
#		* Have train model spit out stats about training loss every 50 epochs
#		* After being trained for n_epochs, store avg_loss and accuracy
#		* Decide whether fix_accuracy should be evaluated on a single pixel or otherwise

 
def trainModel(modelBlock, n_epochs, log_file):
	# Function TRAIN_MODEL
	# Trains all models in modelList for n_epochs
	# Parameters:
	# 		* modelBlock: Nested dictionary of models
	#       * n_epochs: Number of epochs for which to train
	#       * N: size of image to generate
	# Parameters to add
	#		* exp flag that tells whehter or not we are hyperopt or running experiments
	#		* resultBlock
	#		* Number of previously executed epochs

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


	# Read in how many epochs the model has already been trained
	epochs_trained = modelBlock["Meta"]["Epochs_Trained"]
	epochs_total = epochs_trained + n_epochs
	print(epochs_total)


	for epoch in range(n_epochs):

		epoch_real = epoch + epochs_trained
		# Generate training samples and iterate through all models in modelList
		print('Starting epoch %d / %d' % (epoch_real + 1, epochs_total))
		train_set = generateSamples(modelBlock["Meta"]["N"], modelBlock["Meta"]["Distribution"], 50000, test=False)
		for key, val in modelBlock.items():
			if (key != "Meta"):
				runEpoch(modelBlock[key]["Model"], modelBlock["Meta"]["Loss_Function"], modelBlock[key]["Optimizer"], 
					modelBlock["Meta"]["Type"], modelBlock[key]["Batch"], train_set)
		print('Finishing epoch %d / %d' % (epoch_real + 1, epochs_total))
		
		# Want to record test error if the total number of epochs is a multiple 50 or this is the final epoch
		if (((epoch_real % 50) == 0) or (epoch == (n_epochs - 1))):	

			# Every 50 epochs, evaluate the performance of all the models and print summary statistics
			train_set = generateSamples(modelBlock["Meta"]["N"], modelBlock["Meta"]["Distribution"], 100000, test=True)

			loss = []
			accAll = []

			for key, val in modelBlock.items():
				if (key != "Meta"):
					model_accAll, model_loss = checkAccuracy(modelBlock[key]["Model"], 
						modelBlock["Meta"]["Loss_Function"], modelBlock["Meta"]["Type"], modelBlock[key]["Batch"], train_set)
					modelBlock[key]["Loss"] = model_loss
					modelBlock[key]["Acc_All"] = model_accAll
					modelBlock[key]["Acc_Path"] = model_accPath
					modelBlock[key]["Acc_Distract"] = model_accDistract

					loss.append(model_loss)
					accAll.append(model_accAll)


			loss_array = np.asarray(loss)
			accAll_array = np.asarray(accAll)

				
			print('')
			logger.info('Finishing epoch %d / %d' % (epoch_real + 1, epochs_total))
			logger.info('[Loss] Mean:%.6f, Median:%.6f, Best:%.6f' % (np.mean(loss_array),
				np.median(loss_array), np.min(loss_array)))
			logger.info('[Accuracy (All pixels)] Mean:%.6f, Median:%.6f, Best:%.6f ' % (np.mean(accAll_array),
				np.median(accAll_array), np.min(accAll_array)))
			logger.info('')
			print('')

	# Update the total number of epochs trained
	modelBlock["Meta"]["Epochs_Trained"] = epochs_total
	print(modelBlock["Meta"]["Epochs_Trained"])	
	

def trainModel_Exp(modelBlock, resultBlock, n_epochs, log_file, result_file, model_file):
	# Function TRAIN_MODEL
	# Trains all models in modelList for n_epochs
	# Parameters:
	# 		* modelBlock: Nested dictionary of models
	#       * n_epochs: Number of epochs for which to train
	#       * N: size of image to generate
	# Parameters to add
	#		* exp flag that tells whehter or not we are hyperopt or running experiments
	#		* resultBlock
	#		* Number of previously executed epochs

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

	epochs_trained = modelBlock["Meta"]["Epochs_Trained"]
	epochs_total = epochs_trained + n_epochs
	print(epochs_total)

	for epoch in range(n_epochs):

		epoch_real = epoch + epochs_trained
		# Generate training samples and iterate through all models in modelList
		print('Starting epoch %d / %d' % (epoch_real + 1, epochs_total))
		train_set = generateSamples(modelBlock["Meta"]["N"], modelBlock["Meta"]["Distribution"], 50000, test=False)
		
		for key, val in modelBlock.items():
			if (key != "Meta"):
				runEpoch(modelBlock[key]["Model"], modelBlock["Meta"]["Loss_Function"], modelBlock[key]["Optimizer"], 
					modelBlock["Meta"]["Type"], modelBlock[key]["Batch"], train_set)
		print('Finishing epoch %d / %d' % (epoch_real + 1, epochs_total))
		
		# Want to record test error if the total number of epochs is a multiple 50 or this is the final epoch
		if (((epoch_real % 25) == 0) or (epoch == (n_epochs - 1))):	

			# Every 50 epochs, evaluate the performance of all the models and print summary statistics
			train_set = generateSamples(modelBlock["Meta"]["N"], modelBlock["Meta"]["Distribution"], 100000, test=True)

			print('')
			logger.info('Finishing epoch %d / %d' % (epoch_real + 1, epochs_total))


			loss = []
			accAll = []
		
			for key, val in modelBlock.items():
				if (key != "Meta"):
					model_accAll, model_loss = checkAccuracy(modelBlock[key]["Model"], 
						modelBlock["Meta"]["Loss_Function"], modelBlock["Meta"]["Type"], modelBlock[key]["Batch"], train_set)
					modelBlock[key]["Loss"] = model_loss
					modelBlock[key]["Acc_All"] = model_accAll


					resultBlock[key][epoch_real] = {"Loss": model_loss, "Acc_All": model_accAll} 

					loss.append(model_loss)
					accAll.append(model_accAll)

			loss_array = np.asarray(loss)
			accAll_array = np.asarray(accAll)

			logger.info('[Loss] Mean:%.6f, Median:%.6f, Best:%.6f' % (np.mean(loss_array),
				np.median(loss_array), np.min(loss_array)))
			logger.info('[Accuracy (All pixels)] Mean:%.6f, Median:%.6f, Best:%.6f ' % (np.mean(accAll_array),
				np.median(accAll_array), np.min(accAll_array)))
			logger.info('')
			print('')

		# Update the total number of epochs trained
		modelBlock["Meta"]["Epochs_Trained"] = epoch_real
		torch.save(resultBlock, result_file)

		modelBlock_State = convertStateDict(modelBlock)
		torch.save(modelBlock_State, model_file)


def runEpoch(model, loss_fn, optimizer, dtype, batch, train_dset):
	# Function RUN_EPOCH
	# Trains model for one epoch
	# Parameters:
	# 		* model: Pytorch model to train
	#		* train_dset: Training set for model


	loader = DataLoader(train_dset, batch_size=batch, shuffle=True)
	model.train()
	count = 0
	start_time = time.time()
	for x, y in loader:
		x = Variable(x.type(dtype), requires_grad=False)
		if (dtype == torch.cuda.FloatTensor):
			y = Variable(y.type(torch.cuda.LongTensor), requires_grad=False)
		else:
			y = Variable(y.type(torch.LongTensor), requires_grad=False)
		
		# Run the model forward to compute scores and loss.
		output = model(x, dtype).type(dtype)
		loss = loss_fn(output, y).type(dtype)

		# Run the model backward and take a step using the optimizer.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()




def checkAccuracy(model, loss_fn, dtype, batch, test_set):
	# Function CHECK_ACCURACY
	# Evaluate model on test training set
	# Parameters:
	# 		* model: Pytorch model to train
	#		* test_dset: Test set for model


	loader = DataLoader(test_set, batch_size=batch, shuffle=True)

	model.eval()
	num_correct, num_samples = 0, 0
	losses = []

	# The accuracy on all pixels and path pixels can be calculated from the image labels
	# Also record the loss
	for x, label in loader:
		# Cast the image data to the correct type and wrap it in a Variable. At
		# test-time when we do not need to compute gradients, marking the Variable
		# as volatile can reduce memory usage and slightly improve speed.
		x = Variable(x.type(dtype), volatile=True)
		if (dtype == torch.cuda.FloatTensor):
			label = Variable(label.type(torch.cuda.LongTensor), requires_grad=False)
		else:
			label = Variable(label.type(torch.LongTensor), requires_grad=False)

		# Run the model forward and compare with ground truth.
		output = model(x, dtype).type(dtype)
		loss = loss_fn(output, label).type(dtype)


		# Compute Accuracy on PATH pixels
		true_class = label
		pred_class = torch.argmax(output.data, dim = 1)
		num_correct += (true_class == pred_class).sum()
		num_samples += label.size(0)


		losses.append(loss.data.cpu().numpy())


	# Return the fraction of datapoints that were incorrectly classified.
	accAll = 1.0 -  (float(num_correct) / (num_samples))
	avg_loss = sum(losses)/float(len(losses))


	return accAll, avg_loss








