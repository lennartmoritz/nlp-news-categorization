#!/usr/bin/env python3
# Authors: 
# Carlotta Mahncke, Lennart Joshua Moritz, Timon Engelke and Christian Schuler
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from news_categorizer.dateset_tools import CustomNewsDataset, check_dataset_balance, data_merger, data_splitter
from news_categorizer.embeddings import LemmaEmbedding, PretrainedEmbedding, WordEmbedding
from news_categorizer.evaluation import evaluate
from news_categorizer.models import NewsClassifier

# Safe-guarding the script to prevent errors occuring related to multiprocessing
if __name__ == "__main__":
    # Recommended part of the solution (for ubuntu) by encountered error-message
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Accepting arguments to select datasets (classes) for classification task
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)
    input_parser.add_argument('-e', '--embedding', default='lemma', choices=['lemma', 'word', 'pretrained'])
    # Used like:
    # python arg.py -l b t m e      => multiclass 
    # python arg.py -l b t          => binary

    args = input_parser.parse_args()

    task_type = ""  # TODO: Might not be needed anymore(?) Maybe in evaluation-part?

    # Check input for validity
    list_of_classes = args.list
    print('Input parameters:', list_of_classes)
    if len(list_of_classes) == 2:
        task_type = "binary"
        print('Task based on number of classes will be: %s' % (task_type))
    elif len(list_of_classes) > 2:
        task_type = "multiclass"
        print('Task based on number of classes will be: %s' % (task_type))
    elif len(list_of_classes) < 2:
        print('The number of classes is too low for any classification task!')
        sys.exit()

    # Dynamic switch between cpu and gpu (cuda)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    # === DATALOADER ===
    # Selection of data to be used in the model based on input parameter-list

    # Group data for customized data selection
    data_all = pd.read_csv("data/uci-news-aggregator.csv")

    # Preparing datasets based on input parameters (list of classes)
    list_of_datasets = []   # Splittet based on class_key = {e, b, t, m}
    for class_key in list_of_classes:
        data_subset = data_splitter(data_all, 'CATEGORY', class_key)
        list_of_datasets.append(data_subset)

    # Merge dataset to contain all required classes
    data = data_merger(list_of_datasets)

    unique_categories = len(data['CATEGORY'].unique())

    # Get word embeddings based on currently selected dataset
    if args.embedding == 'pretrained':
        print('Using pretrained embedding')
        embedding = PretrainedEmbedding(data)
    elif args.embedding == 'lemma':
        print('Using lemma embedding')
        embedding = LemmaEmbedding(data)
    elif args.embedding == 'word':
        print('Using word embedding')
        embedding = WordEmbedding(data)
    else:
        print('Invalid embedding specified')
        sys.exit()

    # Dataset based on DataLoader and preselected "data" (which categories?)
    dataset = CustomNewsDataset(data, embedding, list_of_classes)

    #train_size = int(0.8 * len(dataset))
    #test_size = len(dataset) - train_size
    train_size = 100 #8192               # Sub-set for quick debugging
    test_size = 100# 1024                # Sub-set for quick debugging    

    validate_size = len(dataset) - (train_size + test_size) # Unused sub-set for quick debugging

    # TODO: check that the datasets are balanced, i.e. all categories must appear
    training_dataset, testing_dataset, validating_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, validate_size])

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
    test_generator = DataLoader(testing_dataset, **params)
    #validate_generator = DataLoader(validating_dataset, **params)

    learning_rate = 0.01    # How fast the model learns
    num_epochs = 5          # How often the model walks through the data

    # Network Parameters
    hidden_size = 100  # 1st layer and 2nd layer number of features
    input_size = embedding.embeddings_len  # Words in vocab
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

    # Tensorboard
    current_classes_as_string = "".join(list_of_classes)
    run_name = str(args.embedding) + "-" + current_classes_as_string + "-" + str(train_size) + "-" + str(num_epochs)
    writer = SummaryWriter(os.path.join('runs', run_name))

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

        # calculate average loss
        avg_loss = avg_loss / total_batch
        # logging
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print('Finished epoch [%d/%d], Average loss: %.5f, Train-Size: %d, Test-Size: %d' % (epoch + 1, num_epochs, avg_loss, train_size, test_size))

    writer.flush()

    # Save model after training TODO: Careful! This gets big really fast!
    save_model = False

    if save_model:
        os.makedirs("./models", exist_ok=True)
        news_net.to("cpu")
        dummy_input = torch.zeros((input_size))
        traced_model = torch.jit.trace(news_net, dummy_input)
        torch.jit.save(traced_model, "./models/news_net.pt")
        my_model =torch.jit.load("./models/news_net.pt")
        news_net.to(device)

    # Calculate Accuracy
    news_net.eval()
    all_predictions = []
    all_labels = []

    # Calculate predictions for the test dataset
    for test_data, labels in tqdm(test_generator):
        test_data, labels = test_data.to(device), labels.to(device)
        all_labels.append(labels.cpu().detach().numpy())
        outputs = news_net(test_data)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.append(predicted.cpu().detach().numpy())

    # Concatenate evaluation predictions from all test batches
    all_predictions = torch.Tensor(np.concatenate(all_predictions))
    all_labels = torch.Tensor(np.concatenate(all_labels))

    evaluate(all_labels, all_predictions, run_name)
