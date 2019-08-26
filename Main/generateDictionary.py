import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkFiles as NF
import numpy as np
import logging

# Module ToDo List
#	Do we want to add weight decay as a hyperparameter?
#	What is a sensible value for the weight_decay
#	Add function to generate different types of masks


def generateDictionary_Hyperopt(N_models, model_type, layers, input_size, hidden_size, image_size, loss_fn, dtype):
	
	# This is the parameters for the distribution of path lengths passed to generate samples
	# n is the longest path we want to generate uniformly over
	n = 25
	distribution = np.ones(n)/n

	# Set up model dictionaries with meta entires that stores key properties of the model
	modelBlock = {"Meta": {"Model_Type": model_type, "Loss_Function": loss_fn, "Layers": layers, 
		"Epochs_Trained": 0, "Type": dtype, "N": image_size, "Distribution": distribution}}
	resultBlock = {}

	# Generate a vector of hyperparameters for the number of models
	lr_vec = np.power(10, (np.random.uniform(-2.5, -5, N_models)))
	weight_decay_vec = np.power(10, (np.random.uniform(-2.5, -5, N_models)))
	batch_size_vec = np.around(np.random.uniform(32, 256, N_models))
	batch_size_vec = batch_size_vec.astype(int)

	for i in range(N_models):
		modelBlock[i] = {}
		modelInit(modelBlock, model_type, i, input_size, hidden_size, layers, image_size)
		modelBlock[i]["Model"].type(dtype)
		modelBlock[i]["Learning"] = lr_vec[i]
		modelBlock[i]["Batch"] = np.asscalar(batch_size_vec[i])
		modelBlock[i]["Weight_Decay"] = np.asscalar(weight_decay_vec[i])
		modelBlock[i]["Optimizer"] = optim.Adam(modelBlock[i]["Model"].parameters(), 
			lr = modelBlock[i]["Learning"], weight_decay = modelBlock[i]["Weight_Decay"])
		modelBlock[i]["Loss"] = 100.0
		modelBlock[i]["Accuracy"] = 1.0

		resultBlock["Meta"] = {"Total_Epochs": 0}
		resultBlock[i] = {"Hyperparameter":{}}
		resultBlock[i]["Hyperparameter"]["Learning"] = lr_vec[i]
		resultBlock[i]["Hyperparameter"]["Batch"] = np.asscalar(batch_size_vec[i])
		resultBlock[i]["Hyperparameter"]["Weight_Decay"] = np.asscalar(batch_size_vec[i])
		resultBlock[i]["Hyperparameter"]["Max_Epoch"] = 0


	return modelBlock, resultBlock


def generateDictionary_Exp(N_models, model_type, layers, input_size, hidden_size, image_size, loss_fn, dtype, hyperparameter):
	
	modelBlock = {}
	resultBlock = {}

	# This is the parameters for the distribution of path lengths passed to generate samples
	# n is the longest path we want to generate uniformly over
	n = 25
	distribution = np.ones(n)/n

	modelBlock["Meta"] = {"Model_Type": model_type, "Epochs_Trained": 0, 
		"Type": dtype, "N": image_size, "Distribution": distribution, "Layers": layers, "Loss_Function": loss_fn,
		"Input": input_size, "Hidden": hidden_size}



	lr = hyperparameter[model_type][layers]["Learning"]
	batch_size = hyperparameter[model_type][layers]["Batch"]
	weight_decay = hyperparameter[model_type][layers]["Weight_Decay"]

	modelBlock["Meta"]["Learning"] = lr
	modelBlock["Meta"]["Batch"] = batch_size
	modelBlock["Meta"]["Weight_Decay"] = weight_decay

	for i in range(N_models):
		modelBlock[i] = {}
		# Note that here we need to pass i in rather than layers
		modelInit(modelBlock, model_type, i, input_size, hidden_size, layers, image_size)
		modelBlock[i]["Model"].type(dtype)
		modelBlock[i]["Learning"] = lr
		modelBlock[i]["Batch"] = batch_size
		modelBlock[i]["Optimizer"] = optim.Adam(modelBlock[i]["Model"].parameters(), lr = lr, weight_decay = weight_decay)
		modelBlock[i]["Loss"] = 100.0
		modelBlock[i]["Accuracy"] = 1.0

		resultBlock[i] = {}

	# Then in the actual code, results get saved as resultBlock[layer][model id][epoch] = {Dictionary of results}

	return modelBlock, resultBlock


def modelInit(modelBlock, model_type, key, input_size, hidden_size, layers, image_size):
	logger = logging.getLogger(__name__)
	if (model_type == "DeepNet"):
		modelBlock[key]["Model"] = NF.DeepNet(input_size, hidden_size, input_size, layers)
	elif (model_type == "DeepNetInput"):
		modelBlock[key]["Model"] = NF.DeepNetInput(input_size, hidden_size, input_size, layers)
	elif (model_type == "Recurrent"):
		modelBlock[key]["Model"] = NF.Recurrent(input_size, hidden_size, input_size, layers)
	elif (model_type == "RecurrentScaled"):
		modelBlock[key]["Model"] = NF.RecurrentScaled(input_size, hidden_size, input_size, layers)
	elif (model_type == "RecurrentMasked5"):
		# This produces a model with a 5 x 5 gird around the pixel
		# The 2 indicated that we want the grid to start two pixels in every direction of a given pixel
		modelBlock[key]["Model"] = NF.RecurrentScaledMasked(input_size, hidden_size, input_size, layers, image_size, 2)
	elif (model_type == "RecurrentGrid"):
		modelBlock[key]["Model"] = NF.RecurrentScaledGrid(input_size, hidden_size, input_size, layers, image_size)
	elif (model_type == "GridFixed"):
		modelBlock[key]["Model"] = NF.RecurrentScaledGridFixed(input_size, hidden_size, input_size, layers, image_size)
	elif (model_type == "RecurrentMultiplicative"):
		modelBlock[key]["Model"] = NF.RecurrentScaledMultiplicative(input_size, hidden_size, input_size, layers)
	else:
		logger.warning('Model type not recognized')


def convertStateDict(modelBlock):

	# The deep copy is important here.  If not done, we end up modifying the original modelBlock
	modelBlock_State = copy.deepcopy(modelBlock)

	for key, val in modelBlock.items():
		if (key != "Meta"):
			model = modelBlock[key]["Model"].state_dict()
			optimizer = modelBlock[key]["Optimizer"].state_dict()
			modelBlock_State[key]["Model"] = model
			modelBlock_State[key]["Optimizer"] = optimizer

	return modelBlock_State


def loadStateDict(modelBlock_State):

	modelBlock = copy.deepcopy(modelBlock_State)
	model_type = modelBlock['Meta']['Model_Type']
	input_size = modelBlock['Meta']['Input']
	hidden_size = modelBlock['Meta']['Hidden']
	layers = modelBlock['Meta']['Layers']
	image_size = modelBlock['Meta']['N']

	lr = modelBlock['Meta']['Learning']
	weight_decay = modelBlock['Meta']['Weight_Decay']

	for key, val in modelBlock.items():
		if (key != "Meta"):
			modelInit(modelBlock, model_type, key, input_size, hidden_size, layers, image_size)
			modelBlock[key]["Optimizer"] = optim.Adam(modelBlock[key]["Model"].parameters(), lr = lr, weight_decay = weight_decay)
			modelBlock[key]["Model"].load_state_dict(modelBlock_State[key]["Model"])
			modelBlock[key]["Optimizer"].load_state_dict(modelBlock_State[key]["Optimizer"])

	return modelBlock










