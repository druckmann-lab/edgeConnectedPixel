import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
from torch.autograd import Variable
from weightMask import generateSquareWeightMask, generateGridWeightMask

## Netowrk Types ##

class Unet(torch.nn.Module):
	def __init__(self):
		super(Unet, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, 3, padding='same')
		self.conv2 = nn.Conv2d(16, 16, 3, padding='same')
		self.pool = nn.MaxPool2d(2, 2)

		self.conv3 = nn.Conv2d(16, 32, 3, padding='same')
		self.conv4 = nn.Conv2d(32, 32, 3, padding='same')

		self.conv5 = nn.Conv2d(32, 64, 3, padding='same')
		self.conv6 = nn.Conv2d(64, 64, 3, padding='same')

		#Expansive path 
		self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
		self.upconv1 = nn.Conv2d(64, 32, 3, padding='same')
		self.conv7 = nn.Conv2d(64, 32, 3, padding='same')
		self.conv8 = nn.Conv2d(32, 32, 3, padding='same')

		self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
		self.upconv2 = nn.Conv2d(32, 16, 3, padding='same')
		self.conv9 = nn.Conv2d(32, 16, 3, padding='same')
		self.conv10 = nn.Conv2d(16, 16, 3, padding='same')

		self.conv11 = nn.Conv2d(16, 1, 1)

	def forward(self, x, dtype):
		x = torch.reshape(x, (32, 16, 16))
		x = x[:, None, :, :]
		#print(x.size())
		c1 = F.relu(self.conv1(x))
		c1 = F.relu(self.conv2(c1))
		c2 = self.pool(c1)

		c2 = F.relu(self.conv3(c2))
		c2 = F.relu(self.conv4(c2))
		c3 = self.pool(c2)

		c3 = F.relu(self.conv5(c3))
		c3 = F.relu(self.conv6(c3))

		c4 = self.upconv1(self.upsample1(c3))
		# Concatenate along the channel dimension
		c4 = torch.cat([c2, c4], 1)
		c4 = F.relu(self.conv7(c4))
		c4 = F.relu(self.conv8(c4))

		c5 = self.upconv2(self.upsample2(c4))
		# Concatenate along the channel dimension
		c5 = torch.cat([c1, c5], 1)
		c5 = F.relu(self.conv9(c5))
		c5 = F.relu(self.conv10(c5))

		output = F.tanh(self.conv11(c5))
		output = torch.reshape(output, (32, 1, 256))
		output = torch.squeeze(output)
		#print(output.size())
		return output


class DeepNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers):
		super(DeepNet, self).__init__()
		d = collections.OrderedDict()

		# First hidden layer:
		d.update({('Layer0', nn.Linear(D_in, H))})
		d.update({('Tanh0', nn.Tanh())})
		
		# Intermediate hidden layers
		for i in range(1,layers):
			d.update({('Layer'+str(i), nn.Linear(H, H))})
			d.update({('Tanh'+str(i), nn.Tanh())})


		self.hiddenLayers = nn.Sequential(d)

		self.output = nn.Linear(H, D_out)
		self.tanh = nn.Tanh()

	def forward(self, x, dtype):
		x = self.hiddenLayers(x)
		y_pred = self.tanh(self.output(x))
		return y_pred


class DeepNetReLU(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers):
		super(DeepNetReLU, self).__init__()
		d = collections.OrderedDict()

		# First hidden layer:
		d.update({('Layer0', nn.Linear(D_in, H))})
		d.update({('ReLU0', nn.ReLU())})
		
		# Intermediate hidden layers
		for i in range(1,layers):
			d.update({('Layer'+str(i), nn.Linear(H, H))})
			d.update({('ReLU'+str(i), nn.ReLU())})


		self.hiddenLayers = nn.Sequential(d)

		self.output = nn.Linear(H, D_out)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, x, dtype):
		x = self.hiddenLayers(x)
		y_pred = self.tanh(self.output(x))
		return y_pred


class DeepNetSkip(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers):
		super(DeepNetSkip, self).__init__()
		d = collections.OrderedDict()
		self.input_size = D_in
		self.hidden_size = H
		self.output_size = D_out
		
		# Intermediate hidden layers
		for i in range(0,layers):
			if (i % 4 == 0):
				d.update({('Layer'+str(i), inputLayers(self.input_size, self.hidden_size))})
			else:	
				d.update({('Layer'+str(i), feedforwardLayer(self.input_size, self.hidden_size))})

		self.hiddenLayers = inputSequential(d)

		self.outputLayer = nn.Linear(self.hidden_size, self.output_size)
		self.tanh = nn.Tanh()

	def forward(self, x, dtype):
		#dtype = torch.cuda.FloatTensor
		u = Variable(torch.zeros(self.hidden_size).type(dtype))
		u = self.hiddenLayers(u, x)
		y_pred = self.tanh(self.outputLayer(u))
		return y_pred


class DeepNetInput(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers):
		super(DeepNetInput, self).__init__()
		d = collections.OrderedDict()
		self.input_size = D_in
		self.hidden_size = H
		self.output_size = D_out
		
		# Intermediate hidden layers
		for i in range(0,layers):
			d.update({('Layer'+str(i), inputLayers(self.input_size, self.hidden_size))})


		self.hiddenLayers = inputSequential(d)

		self.outputLayer = nn.Linear(self.hidden_size, self.output_size)
		self.tanh = nn.Tanh()

	def forward(self, x, dtype):
		#dtype = torch.cuda.FloatTensor
		u = Variable(torch.zeros(self.hidden_size).type(dtype))
		u = self.hiddenLayers(u, x)
		y_pred = self.tanh(self.outputLayer(u))
		return y_pred



class Recurrent(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers):
		super(Recurrent, self).__init__()
		self.iteratedLayer = RepeatedLayers(D_in, H, layers)
		self.outputLayer = nn.Linear(H, D_out)
		self.tanh = nn.Tanh()
		self.hidden_size = H

	def forward(self, x, dtype):
		#dtype = torch.cuda.FloatTensor
		u = Variable(torch.zeros(self.hidden_size).type(dtype))
		u = self.iteratedLayer(u, x)
		y_pred = self.tanh(self.outputLayer(u))
		return y_pred


class RecurrentScaled(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers):
		super(RecurrentScaled, self).__init__()
		self.iteratedLayer = RepeatedLayersScaled(D_in, H, layers)
		self.outputLayer = nn.Linear(H, D_out)
		self.tanh = nn.Tanh()
		self.hidden_size = H

	def forward(self, x, dtype):
		#dtype = torch.cuda.FloatTensor
		u = Variable(torch.zeros(self.hidden_size).type(dtype))
		u = self.iteratedLayer(u, x)
		y_pred = self.tanh(self.outputLayer(u))
		return y_pred

class RecurrentScaledMasked(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers, imageSize, boundarySize):
		super(RecurrentScaledMasked, self).__init__()
		weightMask, diagMask = generateSquareWeightMask(imageSize, boundarySize)
		self.iteratedLayer = RepeatedLayersScaledMasked(D_in, H, layers, weightMask, diagMask)
		self.outputLayer = nn.Linear(H, D_out)
		self.tanh = nn.Tanh()
		self.hidden_size = H

	def forward(self, x, dtype):
		#dtype = torch.cuda.FloatTensor
		u = Variable(-1*torch.ones(self.hidden_size).type(dtype))
		u = self.iteratedLayer(u, x)
		#y_pred = self.tanh(self.outputLayer(u))
		return u


class RecurrentScaledGrid(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers, imageSize):
		super(RecurrentScaledGrid, self).__init__()
		weightMask, diagMask = generateGridWeightMask(imageSize)
		self.iteratedLayer = RepeatedLayersScaledMasked(D_in, H, layers, weightMask, diagMask)
		self.outputLayer = nn.Linear(H, D_out)
		self.tanh = nn.Tanh()
		self.hidden_size = H

	def forward(self, x, dtype):
		#dtype = torch.cuda.FloatTensor
		
		u = Variable(-1*torch.ones(self.hidden_size).type(dtype))
		u = self.iteratedLayer(u, x)
		#y_pred = u#(self.tanh(self.outputLayer(u)))
		return u


class RecurrentScaledGridFixed(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers, imageSize):
		super(RecurrentScaledGridFixed, self).__init__()
		weightMask, diagMask = generateGridWeightMask(imageSize)
		self.iteratedLayer = RepeatedLayersMaskedFixed(D_in, H, layers, weightMask, diagMask)
		self.outputLayer = nn.Linear(H, D_out)
		self.tanh = nn.Tanh()
		self.hidden_size = H

	def forward(self, x, dtype):
		#dtype = torch.cuda.FloatTensor
		
		u = Variable(-1*torch.ones(self.hidden_size).type(dtype))
		u = self.iteratedLayer(u, x)
		#y_pred = u#(self.tanh(self.outputLayer(u)))
		return u




class RecurrentScaledMultiplicative(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers):
		super(RecurrentScaledMultiplicative, self).__init__()
		self.iteratedLayer = RepeatedLayersMultiplicative(D_in, H, layers)
		self.outputLayer = nn.Linear(H, D_out)
		self.tanh = nn.Tanh()
		self.hidden_size = H

	def forward(self, x, dtype):
		#dtype = torch.cuda.FloatTensor
		u = Variable(torch.zeros(self.hidden_size).type(dtype))
		u = self.iteratedLayer(u, x)
		y_pred = self.tanh(self.outputLayer(u))
		return y_pred













## Classes used to construct network types ##

class inputLayers(nn.Module):
	def __init__(self, D_input, hidden):
		super(inputLayers, self).__init__()
		self.hiddenWeight = nn.Linear(hidden, hidden)
		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
		self.tanh = nn.Tanh()

	def forward(self, x, y):
		u = F.tanh(self.hiddenWeight(x) + self.inputWeight(y))
		return u


class feedforwardLayer(nn.Module):
	def __init__(self, D_input, hidden):
		super(feedforwardLayer, self).__init__()
		self.hiddenWeight = nn.Linear(hidden, hidden)
		self.tanh = nn.Tanh()

	def forward(self, x, y):
		u = self.tanh(self.hiddenWeight(x))
		return u


class inputSequential(nn.Sequential):
	def forward(self, inputOne, inputTwo):
		hidden  = inputOne
		for module in self._modules.values():
			hidden = module(hidden, inputTwo)
		return hidden

class RepeatedLayers(torch.nn.Module):
	def __init__(self, D_input, hidden, layers):
		super(RepeatedLayers, self).__init__()
		self.iteration = layers
		self.hiddenWeight = nn.Linear(hidden, hidden)
		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
		self.tanh = nn.Tanh()

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		for _ in range(0, self.iteration):
			v = self.hiddenWeight(u) + self.inputWeight(input)
			u = self.tanh(v)
		return u


class RepeatedLayersScaled(torch.nn.Module):
	def __init__(self, D_input, hidden, layers):
		super(RepeatedLayersScaled, self).__init__()
		self.iteration = layers
		self.hiddenWeight = nn.Linear(hidden, hidden)
		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
		self.tanh = nn.Tanh()
		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		for _ in range(0, self.iteration):
			v = self.hiddenWeight(u) + self.inputWeight(input)
			u = self.tanh(v * self.scalar.expand_as(v))
		return u

class RepeatedLayersMultiplicative(torch.nn.Module):
	def __init__(self, D_input, hidden, layers):
		super(RepeatedLayersMultiplicative, self).__init__()
		self.iteration = layers
		self.hiddenWeight = nn.Linear(hidden, hidden)
		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
		self.tanh = nn.Tanh()
		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		for _ in range(0, self.iteration):
			v = self.hiddenWeight(u)*self.inputWeight(input)
			u = self.tanh(v * self.scalar.expand_as(v))
		return u


class RepeatedLayersMasked(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask):
		super(RepeatedLayersMasked, self).__init__()
		self.iteration = layers
		self.hiddenWeight = nn.Parameter(torch.Tensor(hidden, hidden))
		self.hiddenBias = nn.Parameter(torch.Tensor(hidden))
		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
		self.weightMask = weightMask
		self.tanh = nn.Tanh()

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		for _ in range(0, self.iteration):
			v = F.linear(u, (self.weightMask * self.hiddenWeight), self.hiddenBias) + self.inputWeight(input)
			u = self.tanh(v)
		return u


class RepeatedLayersScaledMasked(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask, diagMask):
		super(RepeatedLayersScaledMasked, self).__init__()

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = ~self.mask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.mask
		self.invertDiag = ~self.diagMask #torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.diagMask
		self.iteration = layers
		self.hiddenWeight = nn.Linear(hidden, hidden)
		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
		self.tanh = nn.Tanh()
		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)
		self.hiddenWeight.weight.data[self.invertMask] = 0
		#self.hiddenWeight.weight.data[self.mask] = 0.25

		self.inputWeight.weight.data[self.invertDiag] = 0
		#self.inputWeight.weight.data[self.diagMask] = 1
		#self.hiddenWeight.bias.data[:] = -0.15

		self.hiddenWeight.weight.register_hook(self.backward_hook)
		self.inputWeight.weight.register_hook(self.backward_hook_input)
		

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		for _ in range(0, self.iteration):
			v = self.hiddenWeight(u) + self.inputWeight(input)
			u = self.tanh(v * self.scalar.expand_as(v))
			#u = torch.sign(u)
		return u

	def backward_hook(self, grad):
		out = grad.clone()
		out[self.invertMask] = 0
		return out


	def backward_hook_input(self, grad):
		out = grad.clone()
		out[self.invertDiag] = 0
		return out


# This is the class to use to get forward-engineered values
class RepeatedLayersMaskedFixed(torch.nn.Module):
	def __init__(self, D_input, hidden, layers, weightMask, diagMask):
		super(RepeatedLayersMaskedFixed, self).__init__()

		self.mask = weightMask
		self.diagMask = diagMask
		self.invertMask = torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.mask
		self.invertDiag = torch.ones((hidden, hidden)).type(torch.ByteTensor) - self.diagMask
		self.iteration = layers
		self.hiddenWeight = nn.Linear(hidden, hidden)
		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
		self.tanh = nn.Tanh()
		self.scalar = nn.Parameter(torch.ones(1)*20, requires_grad=False)
		self.hiddenWeight.weight.data[self.invertMask] = 0
		self.hiddenWeight.weight.data[self.mask] = 0.25

		self.inputWeight.weight.data[:] = 0
		self.inputWeight.weight.data[self.diagMask] = 1
		self.hiddenWeight.bias.data[:] = -0.15

		# Turn off the gradient on all parameters
		for param in self.hiddenWeight.parameters():
			param.requires_grad = False

		for param in self.inputWeight.parameters():
			param.requires_grad = False
		

	def forward(self, initial_hidden, input):
		u = initial_hidden.clone()
		for _ in range(0, self.iteration):
			v = self.hiddenWeight(u) + self.inputWeight(input)
			u = self.tanh(v * self.scalar.expand_as(v))
			#u = torch.sign(u)
		return u









