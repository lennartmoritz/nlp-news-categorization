#!/usr/bin/env python3
# Authors: 
# Carlotta Mahncke, Lennart Joshua Moritz, Timon Engelke and Christian Schuler

from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy

# TODO: Read model and data for evaluation
def read_model_and_data(model_name):
    current_model = torch.load("./models/" + model_name)
    return current_model


# TODO: Create evaluations
def evaluate_model(loaded_model):
    loaded_model.eval()
    return "It worked."


# TODO: Save evaluations to file for visualisation
def safe_evaluations_to_file():
    pass


# TODO: Included this because it seems like we are supposed to define the model prior to loading
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
# Define the network
class NewsClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
            self.relu = nn.ReLU()
            self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

        # Accept input and return an output
        def forward(self, x):
            out = self.layer_1(x)
            out = self.relu(out)
            out = self.layer_2(out)
            out = self.relu(out)
            out = self.output_layer(out)
            return out

# Safe-guarding the script to prevent errors occuring related to multiprocessing
if __name__ == "__main__":
    # Recommended part of the solution (for ubuntu) by encountered error-message
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Dynamic switch between cpu and gpu (cuda)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)
    
    current_model = NewsClassifier() # TODO: Just defining it like this is not enough!
    # => TypeError: __init__() missing 3 required positional arguments: 'input_size', 'hidden_size', and 'num_classes'
    # => So, how are we supposed to load a model if we have to define it beforehand?
    # ==> Sometimes the model will have more, sometimes fewer words in the embedding and each binary-classifier will require its own model definition?!
    # ===> That just sounds wrong...
    current_model = torch.load("./models/currentModel")
    current_model.eval()
    
    # Load prior trained model
    #current_model = read_model_and_data("currentModel")
    
    # Evaluate the loaded model
    #status = evaluate_model(current_model)
    
    # Calculate Recall
    # TODO: For binary classification
    # TODO: For multi-class classification
    
    # Calculate Precision
    # TODO: For binary classification
    # TODO: For multi-class classification
        
    # Calculate F1-Scores
    # TODO: For binary classification
    # TODO: For multi-class classification    
    
    
    
