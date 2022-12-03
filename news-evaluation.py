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
def read_model_and_data():
    pass


# TODO: Create evaluations
def evaluate_model():
    pass


# TODO: Save evaluations to file for visualisation
def safe_evaluations_to_file():
    pass


# Safe-guarding the script to prevent errors occuring related to multiprocessing
if __name__ == "__main__":
    # Recommended part of the solution (for ubuntu) by encountered error-message
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Dynamic switch between cpu and gpu (cuda)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)
    
    # Load prior trained model
    
    
    # Calculate Recall
    # TODO: For binary classification
    # TODO: For multi-class classification
    
    # Calculate Precision
    # TODO: For binary classification
    # TODO: For multi-class classification
        
    # Calculate F1-Scores
    # TODO: For binary classification
    # TODO: For multi-class classification    
    
    
    
