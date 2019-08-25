import torch
from generateDictionary import loadStateDict

model_file = '../../EdgePixel_Results/Experiments/ResultBlock/modelBlock_test.pth.tar'
modelBlock_state = torch.load(model_file)

modelBlock = loadStateDict(modelBlock_state)