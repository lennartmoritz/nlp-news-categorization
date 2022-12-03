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
from torchmetrics import Accuracy, Recall, Precision, F1Score
import sys


# Split the dataset into sub-set of specific category
def data_splitter(dataset, key, value) -> pd.DataFrame:
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

# Generate the word embeddings for the selected dataset
def get_word_embeddings(dataset):
    """
    Returns:
    word_embeddings:    A dict that maps word names as keys to an automatically generated word_id
    """
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
    data_all = pd.read_csv("data/uci-news-aggregator.csv")
    
    #grouped_dataset = data_all.groupby(data_all['CATEGORY'])
    data_e = data_splitter(data_all, 'CATEGORY', 'e') #grouped_dataset.get_group('e')
    data_b = data_splitter(data_all, 'CATEGORY', 'b')
    data_t = data_splitter(data_all, 'CATEGORY', 't')
    data_m = data_splitter(data_all, 'CATEGORY', 'm')
    
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

    # Selection of data to be used in the model
    # TODO: Actual selection happening => This might include some restructuring of above data #PythonMagic
    data = data_all                     # | data_eb     | data_et       | ...
    unique_categories = len(data['CATEGORY'].unique())

    categories = ['e', 'b', 't', 'm']   # | ['e', 'b']  | ['e', 't']    | ...
    
    # Get word embeddings based on currently selected dataset
    embeddings = get_word_embeddings(data)

    # Dataset based on DataLoader and preselected "data" (which categories?)
    dataset = CustomNewsDataset(data, embeddings, categories)
    
    #train_size = int(0.8 * len(dataset))
    train_size = 500 # Sub-set for quick debugging
    #test_size = len(dataset) - train_size
    test_size = 100  # Sub-set for quick debugging    
    
    validate_size = len(dataset) - (train_size + test_size) # Unused sub-set for quick debugging
    
    #TODO: check that the datasets are balanced, i.e. all categories must appear
    training_dataset, testing_dataset, validating_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, validate_size])
    print("Type of training_dataset: ", type(training_dataset))
    
    def check_dataset_balance(this_dataset, intended_num_of_classes) -> bool:
        """
        Returns True if intended number of unique classes are present in this dataset
        """
        print("Checking that all unique classes are represented...")
        duplicate_removal_set = set()
        for item in tqdm(this_dataset):
            duplicate_removal_set.add(item[1])
        return len(duplicate_removal_set) == intended_num_of_classes

    if False:
        assert check_dataset_balance(training_dataset, unique_categories)
        assert check_dataset_balance(testing_dataset, unique_categories)
        assert check_dataset_balance(validating_dataset, unique_categories)

    
    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}

    # Dataloader
    train_generator = DataLoader(training_dataset, **params)
    print("Type of train_generator: ", type(train_generator))
    test_generator = DataLoader(testing_dataset, **params)
    #validate_generator = DataLoader(validating_dataset, **params)

    learning_rate = 0.01    # How fast the model learns
    num_epochs = 5          # How often the model walks through the data

    # Network Parameters
    hidden_size = 100  # 1st layer and 2nd layer number of features
    input_size = len(embeddings)  # Words in vocab
    num_classes = unique_categories  # Categories: "e", "b", "t", "m"
    # e: entertainment | b: business | t: science and technology | m: health
    # e: 152469        | b: 115967   | t: 108344                 | m: 45639

    # Model for news classification
    news_net = NewsClassifier(input_size, hidden_size, num_classes)
    news_net = news_net.to(device)
    news_net.train()
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  # This includes the Softmax loss function
    optimizer = torch.optim.Adam(news_net.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        # Determine the number of min-batches based on the batch size and size of training data
        avg_loss = 0
        total_batch = len(train_generator)
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

    ## Specificity
    from sklearn.metrics import confusion_matrix
    resulting_confusion_matrix = confusion_matrix(all_labels, all_predictions)
    FP = resulting_confusion_matrix.sum(axis=0) - np.diag(resulting_confusion_matrix)  
    FN = resulting_confusion_matrix.sum(axis=1) - np.diag(resulting_confusion_matrix)
    TP = np.diag(resulting_confusion_matrix)
    TN = resulting_confusion_matrix.sum() - (FP + FN + TP)

    FP = sum(FP)
    FN = sum(FN)
    TP = sum(TP)
    TN = sum(TN)

    print(f"fp: {FP}\tfn: {FN}\ttp: {TP} \ttn{TN}")

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print(f"Sensitivity, hit rate, recall, or true positive rate: {TPR}")
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print(f"Specificity or true negative rate: {TNR}")
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print(f"Precision or positive predictive value: {PPV}")
    # Negative predictive value
    NPV = TN/(TN+FN)
    print(f"Negative predictive value: {NPV}")
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print(f"Overall accuracy: {ACC}")    
