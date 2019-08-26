# Author: Brett Larsen
# Code for generating samples of edge-connected pixels along with image labels
#	generateSamplesFull.py returns image and label of edge connected pixels
#	Use generateSamplesPropagate.py to generate images in the process of propagating labels


import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt


def generateSamples(N, distribution, numTraining, test=False):
	# Key parameters for generating samples
						# Image size is N x N
	p = 0.05				# Probability of seed pixels

	N2 = N**2
	min_length = 0

	center = np.ceil(N2/2).astype(int)

	quota = np.ceil(distribution*numTraining)
	quota = quota.astype(int)
	maxLength = len(quota)
	current = np.zeros(maxLength)
	trainFeatures = np.zeros((numTraining, N2))
	trainLabels = np.zeros((numTraining))
	pathLengths = np.zeros((numTraining))

	j = 0

	# Generate training images and labels
	while(j < numTraining):
		# Code for generating random lines that occasionally touch an edge
		X = np.random.binomial(1, p, (N, N))

		row, col = np.nonzero(X)
		l = len(row)

		path = 1
		

		for i in range(0, l):
			k = 1
			row_current = row[i]
			col_current = col[i]
			direction = int(np.floor(np.random.uniform(low = 0.0, high = 4.0)))
			if((row_current == 0) and (col_current != 0) and (col_current != N-1)):
				row_current = row_current + 1
				col_current = col_current
				X[row_current, col_current] = 1
				k = k+1
			elif((row_current == N-1) and (col_current != 0) and (col_current != N-1)):
				row_current = row_current - 1
				col_current = col_current
				X[row_current, col_current] = 1
				k = k+1
			elif((col_current == 0) and (row_current != 0) and (row_current != N-1)):
				row_current = row_current
				col_current = col_current+1
				X[row_current, col_current] = 1
				k = k+1
			elif((col_current == N-1) and (row_current != 0) and (row_current != N-1)):
				row_current = row_current
				col_current = col_current-1
				X[row_current, col_current] = 1
				k = k+1
			while((row_current != 0) and (row_current != N-1) and (col_current != 0) and (col_current != N-1)):
				q = np.random.uniform()
				if ((q > 0.65) and (q <= 0.775)):
					direction = (direction + 1) % 4
				elif ((q > 0.775) and (q <= 0.90)):
					direction = (direction - 1) % 4
				elif ((q > 0.90) and (k > min_length)):
					break
				if(direction == 0):
					row_current = row_current
					col_current = col_current + 1
					X[row_current, col_current] = 1
				elif(direction == 1):
					row_current = row_current + 1
					col_current = col_current
					X[row_current, col_current] = 1
				elif(direction == 2):
					row_current = row_current
					col_current = col_current - 1
					X[row_current, col_current] = 1
				elif(direction == 3):
					row_current = row_current - 1
					col_current = col_current
					X[row_current, col_current] = 1
				k = k+1
			path = np.maximum(path, k)


		# Code for making longer distractors
		xLabel = np.zeros((N,N))

		row, col = np.nonzero(X)
		l = len(row)
		rowNext = []
		colNext = []

		for i in range(0, l):
			if ((row[i] ==0) or (col[i] == 0) or (row[i] == N-1) or (col[i]) == N-1):
				xLabel[row[i], col[i]] = 1
				rowNext.append(row[i])
				colNext.append(col[i])

		diff = 10
		x = 0

		cutoff = np.random.randint(2)

		row = []
		col = []

		while((diff != 0) and (x <= cutoff)):
			xLabel_Old = xLabel[:,:]
			del row[:]
			del col[:]
			row = rowNext[:]
			col = colNext[:]
			l = len(row)
			del rowNext[:]
			del colNext[:]
			if (x == cutoff):
				for i in range(0, l):
					row_current = row[i]
					col_current = col[i]
					q = np.random.uniform()
					if (q < 0.8):
						X[row_current, col_current] = 0
					
			else:
				for i in range(0, l):
					row_current = row[i]
					col_current = col[i]
					if ((row_current != 0) and (X[(row_current - 1), col_current] == 1) and (xLabel[(row_current - 1), col_current] == 0)):
						#xLabel[row_current - 1, col[i]] = 1
						rowNext.append(row_current - 1)
						colNext.append(col_current)
					if ((row_current != N-1) and (X[row_current + 1, col_current] == 1) and (xLabel[(row_current + 1), col_current] == 0)):
						#xLabel[row_current + 1, col_current] = 1
						rowNext.append(row_current + 1)
						colNext.append(col_current)
					if ((col_current != 0) and (X[row_current, col_current-1] == 1) and (xLabel[row_current, col_current-1] == 0)):
						#xLabel[row_current, col_current-1] = 1
						rowNext.append(row_current)
						colNext.append(col_current-1)
					if ((col_current != N-1) and (X[row_current, col_current+1] == 1) and (xLabel[row_current, col_current+1] == 0)):
						#xLabel[row_current, col_current+1] = 1
						rowNext.append(row_current)
						colNext.append(col_current+1)
			diff = len(rowNext)
			x = x+1		

		# Loop for labelling the images and determining path length
		# Code for labelling images
		xLabel = np.zeros((N,N))

		row, col = np.nonzero(X)
		l = len(row)
		rowNext = []
		colNext = []

		for i in range(0, l):
			if ((row[i] ==0) or (col[i] == 0) or (row[i] == N-1) or (col[i]) == N-1):
				xLabel[row[i], col[i]] = 1
				rowNext.append(row[i])
				colNext.append(col[i])

		diff = 10
		x = 0


		row = []
		col = []

		while(diff != 0):
			xLabel_Old = xLabel[:,:]
			del row[:]
			del col[:]
			row = rowNext[:]
			col = colNext[:]
			l = len(row)
			del rowNext[:]
			del colNext[:]
			for i in range(0, l):
				row_current = row[i]
				col_current = col[i]
				if ((row_current != 0) and (X[(row_current - 1), col_current] == 1) and (xLabel[(row_current - 1), col_current] == 0)):
					xLabel[row_current - 1, col[i]] = 1
					rowNext.append(row_current - 1)
					colNext.append(col_current)
				if ((row_current != N-1) and (X[row_current + 1, col_current] == 1) and (xLabel[(row_current + 1), col_current] == 0)):
					xLabel[row_current + 1, col_current] = 1
					rowNext.append(row_current + 1)
					colNext.append(col_current)
				if ((col_current != 0) and (X[row_current, col_current-1] == 1) and (xLabel[row_current, col_current-1] == 0)):
					xLabel[row_current, col_current-1] = 1
					rowNext.append(row_current)
					colNext.append(col_current-1)
				if ((col_current != N-1) and (X[row_current, col_current+1] == 1) and (xLabel[row_current, col_current+1] == 0)):
					xLabel[row_current, col_current+1] = 1
					rowNext.append(row_current)
					colNext.append(col_current+1)
			diff = len(rowNext)
			x = x+1

		if ((x < maxLength) and (current[x - 1] < quota[x-1])):
			pathLengths[j] = x
			X_vec = np.reshape(X, (1,N2))
			xLabel_vec = np.reshape(xLabel, (1,N2))
			xDistract_vec = X_vec - xLabel_vec
			X_vec[X_vec == 0] = -1
			xLabel_vec[xLabel_vec == 0] = -1
			xDistract_vec[xDistract_vec == 0] = -1

			trainFeatures[j, :] = X_vec
			if (xLabel_vec[:, center] == -1):
				trainLabels[j] = 0
			else:
				trainLabels[j] = 1

			current[x-1] = current[x-1]+1
			j = j+1
		elif (x >= maxLength):
			pathLengths[j] = x
			X_vec = np.reshape(X, (1,N2))
			xLabel_vec = np.reshape(xLabel, (1,N2))
			xDistract_vec = X_vec - xLabel_vec
			X_vec[X_vec == 0] = -1
			xLabel_vec[xLabel_vec == 0] = -1
			xDistract_vec[xDistract_vec == 0] = -1

			trainFeatures[j, :] = X_vec
			if (xLabel_vec[:, center] == -1):
				trainLabels[j] = 0
			else:
				trainLabels[j] = 1
			
			current[maxLength-1] = current[maxLength-1]+1
			j = j+1



	if (test):
		trainFeatures = torch.from_numpy(trainFeatures)
		trainLabels = torch.from_numpy(trainLabels)

		trainset = torch.utils.data.TensorDataset(trainFeatures, trainLabels)

		return trainset


	else:
		trainFeatures = torch.from_numpy(trainFeatures)
		trainLabels = torch.from_numpy(trainLabels)

		trainset = torch.utils.data.TensorDataset(trainFeatures, trainLabels)

		return trainset

