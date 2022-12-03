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
from torchmetrics import Recall
from torchmetrics import Precision
from torchmetrics import F1Score


# Split the dataset into sub-set of specific category
def data_splitter(dataset, key, value):
    grouped_dataset = dataset.groupby(dataset[key])
    sub_set = grouped_dataset.get_group(value)
    return sub_set

# Merge the sub-datasets into dataset for the model
def data_merger(list_of_datasets):
    # Concatenate the dataframes
    dataframes = []
    for dataframe in list_of_datasets:
        dataframes.append(dataframe)
    dataframes_merged = pd.concat(dataframes)
    return dataframes_merged

# TODO: Might play a part in the "Cluster-Fuck-Land"-Issue...
# Separate dataset into train-data and test-data (Manual version without Loader)
def get_train_test_data(dataset, fraction):
    train_dataset = dataset.sample(frac=fraction, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset, test_dataset

# Generate the word embeddings for the selected dataset
def get_word_embeddings(dataset):
        # Getting all the vocabularies and indexing to a unique position
        vocab = Counter()
        # Indexing words from the training data
        for text in dataset['TITLE']:
            for word in text.split(' '):
                vocab[word.lower()] += 1

        word_embeddings = {}
        for i, word in enumerate(vocab):
            word_embeddings[word.lower()] = i

        return word_embeddings

# 
class CustomNewsDataset(Dataset):
        def __init__(self, data, embeddings, categories):
            self.labels = data['CATEGORY']
            self.titles = data['TITLE']
            self.embeddings = embeddings
            self.categories = categories

        def __len__(self):
            return len(self.titles)

        def __getitem__(self, idx):
            title = self.titles.iloc[idx]
            embedded_title = np.zeros(len(self.embeddings), dtype=np.float64)

            for word in title.split(' '):
                embedded_title[self.embeddings[word.lower()]] += 1

            label = self.labels.iloc[idx]
            label = self.categories.index(label)
            return torch.Tensor(embedded_title), label
            
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

    # === DATALOADER ===

    # +++ Binary classifications +++ (all posssible combinations of two)
    # Group data for customized data selection
    data_all = pd.read_csv("data/uci-news-aggregator.csv") #TODO: Christian-Question: Loading again to not mess with "data" by grouping on it -> Does that make sense/is it necessary?
    print("Type of data_all: ", type(data_all))
    
    #grouped_dataset = data_all.groupby(data_all['CATEGORY'])
    data_e = data_splitter(data_all, 'CATEGORY', 'e') #grouped_dataset.get_group('e')
    data_b = data_splitter(data_all, 'CATEGORY', 'b')
    data_t = data_splitter(data_all, 'CATEGORY', 't')
    data_m = data_splitter(data_all, 'CATEGORY', 'm')
    print("Type of data_e: ", type(data_e))
    
    # Sanity-Check
    print("Number of e: " + str(len(data_e)) + "\tNumber of b: " + str(len(data_b)) + "\tNumber of t: " + str(len(data_t)) + "\tNumber of m: " + str(len(data_m)))
    
    # Datasets containing only two categories (for binary classification)
    data_eb = data_merger([data_e, data_b]) # Binary (e & b)
    data_et = data_merger([data_e, data_t]) # Binary (e & t)
    data_em = data_merger([data_e, data_m]) # Binary (e & m)
    data_bt = data_merger([data_b, data_t]) # Binary (b & t)
    data_bm = data_merger([data_b, data_m]) # Binary (b & m)
    data_tm = data_merger([data_t, data_m]) # Binary (t & m)

    # Sanity-Check
    #print("Number of eb: " + str(len(data_eb)) + "\tNumber of et: " + str(len(data_et)) + "\tNumber of em: " + str(len(data_em)))

    # +++ Multi-class +++ (all four of them together)
    data_all = data_merger([data_e, data_b, data_t, data_m]) # All (e, b, t, m)
    print("Type of data_all: ", type(data_all))



    ######################################################################## Cluster-Fuck-Land
    # TEMP to not interrupt other coder!
    # !!! data_splitter() is not the same as previously !!!
    full_train_dataset, full_test_dataset = get_train_test_data(data_all, 0.8)
    train_dataset = full_train_dataset
    print("Type of train_dataset: ", type(train_dataset))
    test_dataset = full_test_dataset
    ######################################################################## Cluster-Fuck-Land




    # Selection of data to be used in the model
    # TODO: Actual selection happening => This might include some restructuring of above data #PythonMagic
    data = data_all                     # | data_eb     | data_et       | ...
    print("Type of data: ", type(data))
    categories = ['e', 'b', 't', 'm']   # | ['e', 'b']  | ['e', 't']    | ...
    
    # Get word embeddings based on currently selected dataset
    embeddings = get_word_embeddings(data)


    # Dataset based on DataLoader and preselected "data" (which categories?)
    dataset = CustomNewsDataset(data, embeddings, categories)
    print("Type of dataset: ", type(dataset))
    
    #train_size = int(0.8 * len(dataset))
    train_size = 500 # Sub-set for quick debugging
        
    #test_size = len(dataset) - train_size
    test_size = 100  # Sub-set for quick debugging    
    
    validate_size = len(dataset) - (train_size + test_size) # Unused sub-set for quick debugging
    
    training_dataset, testing_dataset, validating_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, validate_size])
    print("Type of training_dataset: ", type(training_dataset))

    ######################################################################## Cluster-Fuck-Land
    #train_dataset = training_dataset
    #print("Type of train_dataset: ", type(train_dataset))
    #test_dataset = testing_dataset
    ######################################################################## Cluster-Fuck-Land
    
    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}

    # Dataloader
    train_generator = DataLoader(training_dataset, **params)
    print("Type of train_generator: ", type(train_generator))
    test_generator = DataLoader(testing_dataset, **params)
    #validate_generator = DataLoader(validating_dataset, **params)

    # Parameters TODO: Christian-Question: How come we have batch_size in params and then here again?
    learning_rate = 0.01    # How fast the model learns
    num_epochs = 5          # How often the model walks through the data
    batch_size = 150        # How big the groups of separated data entries are
    display_step = 1        # TODO: Is this even used? Also: What is it?

    # Network Parameters
    hidden_size = 100  # 1st layer and 2nd layer number of features
    input_size = len(embeddings)  # Words in vocab
    num_classes = 4  # Categories: "e", "b", "t", "m"
    # e: entertainment | b: business | t: science and technology | m: health
    # e: 152469        | b: 115967   | t: 108344                 | m: 45639

    # Model for news classification
    news_net = NewsClassifier(input_size, hidden_size, num_classes)
    news_net = news_net.to(device)
    news_net.train()
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  # This includes the Softmax loss function
    optimizer = torch.optim.Adam(news_net.parameters(), lr=learning_rate)

    print("Type of train_dataset: ", type(train_dataset))

    # Train the Model
    for epoch in range(num_epochs):
        # Shuffle the data
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True) # TODO: Christian-Question: Why do we use "train_dataset" and not "training_dataset"? 
        
        
        ######################################################################## Cluster-Fuck-Land
        # TODO: trying with training_dataset since error states: AttributeError: 'Subset' object has no attribute 'sample'
        #train_dataset = training_dataset.sample(frac=1).reset_index(drop=True) 
        #
        #train_dataset = training_dataset.sample(frac=1).reset_index(drop=True)
        #
        #print("Type of training_dataset: ", type(training_dataset))
        #training_dataset = training_dataset.sample(frac=1).reset_index(drop=True)
        #
        # total_batch = int(len(training_dataset) / batch_size)
        ######################################################################## Cluster-Fuck-Land
       
        
        # Determine the number of min-batches based on the batch size and size of training data
        total_batch = int(len(train_dataset) / batch_size)
        avg_loss = 0
        # Loop over all batches
        for titles, labels in tqdm(train_generator):
            titles, labels = titles.to(device), labels.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # Zero the gradient buffer
            outputs = news_net(titles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        # 
        avg_loss = avg_loss / total_batch
        print('Finished epoch [%d/%d], Average loss: %.5f, Train-Size: %d, Test-Size: %d' % (epoch + 1, num_epochs, avg_loss, train_size, test_size))

    # Save model after training
    # TODO: Careful! This gets big really fast!
    #torch.save(news_net, "./models/currentModel")
    

    
    ############################################################################
    # TODO: Move evaluation parts to new script [news_evaluation.py]
    
    # Calculate Accuracy
    news_net.eval()
    all_predictions = []
    all_labels = []

    # 
    for test_data, labels in tqdm(test_generator):
        test_data, labels = test_data.to(device), labels.to(device)
        all_labels.append(labels.cpu().detach().numpy())
        outputs = news_net(test_data)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.append(predicted.cpu().detach().numpy())

    # 
    all_predictions = torch.Tensor(np.concatenate(all_predictions))
    all_labels = torch.Tensor(np.concatenate(all_labels))

    labels_unique = (torch.unique(all_labels)).tolist()
    labels_number = len(labels_unique)
    print('Number of labels/classes: %d - which are %s' % (labels_number, labels_unique))
    # TODO: We lose the information about the category (by indexing?) 
    # => Have to track the real categories for later processing


    # Torchmetrics required by task description
    # https://torchmetrics.readthedocs.io/en/stable/classification/recall.html
    # https://torchmetrics.readthedocs.io/en/stable/classification/precision.html
    # https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html

    # Accuracy for binary classification # TODO: Accuracy is not necessary
    
    # Accuracy for multi-class classification
    accuracy = Accuracy(task='multiclass', num_classes=labels_number)
    acc = accuracy(all_predictions, all_labels)
    print('Accuracy of the model on the test data:  %f' % (acc))
    
    # Calculate Recall
    # TODO: For binary classification
    # TODO: For multi-class classification
    recall = Recall(task='multiclass', num_classes=labels_number)
    rec = recall(all_predictions, all_labels)
    print('Recall of the model on the test data: \t %f' % (rec))
    
    recallMacro = Recall(task='multiclass', average='macro', num_classes=labels_number)
    rec = recallMacro(all_predictions, all_labels)
    print('Recall MACRO of the model on the test data: \t %f' % (rec))
    
    recallMicro = Recall(task='multiclass', average='micro', num_classes=labels_number)
    rec = recallMicro(all_predictions, all_labels)
    print('Recall MICRO of the model on the test data: \t %f' % (rec))
    
    # Calculate Precision
    # TODO: For binary classification
    # TODO: For multi-class classification
    precision = Precision(task='multiclass', num_classes=labels_number)
    pre = precision(all_predictions, all_labels)
    print('Precision of the model on the test data: %f' % (pre))
    
    precisionMacro = Precision(task='multiclass', average='macro', num_classes=labels_number)
    pre = precisionMacro(all_predictions, all_labels)
    print('Precision MACRO of the model on the test data:\t %f' % (pre))
    
    precisionMicro = Precision(task='multiclass', average='micro', num_classes=labels_number)
    pre = precisionMicro(all_predictions, all_labels)
    print('Precision MICRO of the model on the test data:\t %f' % (pre))
        
    # Calculate F1-Scores
    # TODO: For binary classification
    # TODO: For multi-class classification    
    f1score = F1Score(task='multiclass', num_classes=labels_number)
    f1s = f1score(all_predictions, all_labels)
    print('F1-Score of the model on the test data:\t %f' % (f1s))

    f1scoreMacro = F1Score(task='multiclass', average='macro', num_classes=labels_number)
    f1s = f1scoreMacro(all_predictions, all_labels)
    print('F1-Score MACRO of the model on the test data:\t %f' % (f1s))
    
    f1scoreMicro = F1Score(task='multiclass', average='micro', num_classes=labels_number)
    f1s = f1scoreMicro(all_predictions, all_labels)
    print('F1-Score MICRO of the model on the test data:\t %f' % (f1s))
    
    f1scoreWeighted = F1Score(task='multiclass', average='weighted', num_classes=labels_number)
    f1s = f1scoreWeighted(all_predictions, all_labels)
    print('F1-Score WEIGHTED of the model on the test data: %f' % (f1s))
    
    #f1scoreNone = F1Score(task='multiclass', average='none', num_classes=labels_number)
    #f1s = f1scoreNone(all_predictions, all_labels)
    #print('F1-Score NONE of the model on the test data:\t %f' % (f1s))
    # TODO: average='none' results in
    # ValueError: only one element tensors can be converted to Python scalars
    #
    # Documentation states:
    # average ... Defines the reduction that is applied over labels. Should be one of the following:
    # micro: Sum statistics over all labels
    # macro: Calculate statistics for each label and average them
    # weighted: Calculates statistics for each label and computes weighted average using their support
    # "none" or None: Calculates statistic for each label and applies no reduction
    #
    
