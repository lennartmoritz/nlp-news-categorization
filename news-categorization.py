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

from news_categorizer.dataset_tools import CustomNewsDataset, data_merger, data_splitter
from news_categorizer.embeddings import LemmaEmbedding, GloveEmbedding, Word2VecEmbedding, WordEmbedding
from news_categorizer.evaluation import evaluate
from news_categorizer.models import NewsClassifier

# Safe-guarding the script to prevent errors occurring related to multiprocessing
if __name__ == "__main__":
    # Recommended part of the solution (for ubuntu) by encountered error-message
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Accepting arguments to select datasets (classes) for classification task
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)
    input_parser.add_argument('-e', '--embedding', default='lemma', choices=['lemma', 'word', 'glove', 'word2vec', 'pretrained'])
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
    # Drop data containing NaN values (two rows are tab unparsable)
    data = data.dropna()

    unique_categories = len(data['CATEGORY'].unique())

    # Get word embeddings based on currently selected dataset
    if args.embedding == 'glove':
        print('Using pretrained glove embedding')
        embedding = GloveEmbedding(data)
    elif args.embedding == 'word2vec' or args.embedding == 'pretrained':
        print('Using pretrained word2vec embedding')
        embedding = Word2VecEmbedding(data)
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

    # split dataset 80:20 into training and test, split training again into training and validation
    train_size = int(0.8 * 0.8 * len(dataset))
    validation_size = int(0.8 * 0.2 * len(dataset))
    test_size = len(dataset) - train_size - validation_size

    # random split ensures that all categories are present in both datasets
    training_dataset, testing_dataset, validating_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}

    # Dataloader
    train_generator = DataLoader(training_dataset, **params)
    validation_generator = DataLoader(validating_dataset, **params)
    test_generator = DataLoader(testing_dataset, **params)

    learning_rate = 0.01    # How fast the model learns
    num_epochs = 10          # How often the model walks through the data

    # Network Parameters
    hidden_size = 100  # 1st layer and 2nd layer number of features
    input_size = embedding.embeddings_len  # Words in vocab
    num_classes = unique_categories  # Categories: "e", "b", "t", "m"
    # e: entertainment | b: business | t: science and technology | m: health
    # e: 152469        | b: 115967   | t: 108344                 | m: 45639

    # Model for news classification
    news_net = NewsClassifier(input_size, hidden_size, num_classes)
    news_net = news_net.to(device)

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
        avg_training_loss = 0
        total = 0
        correct = 0
        total_batch = len(train_generator)
        # Loop over all batches
        news_net.train()
        for titles, labels in tqdm(train_generator):
            titles, labels = titles.to(device), labels.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # Zero the gradient buffer
            outputs = news_net(titles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            avg_training_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # calculate average loss
        avg_training_loss = avg_training_loss / total_batch
        accuracy = correct / total
        writer.add_scalar('train/loss', avg_training_loss, epoch)
        writer.add_scalar('train/accuracy', accuracy, epoch)

        # validation
        avg_validation_loss = 0
        total = 0
        correct = 0
        validation_batch = len(validation_generator)
        news_net.eval()
        for titles, labels in tqdm(validation_generator):
            titles, labels = titles.to(device), labels.to(device)
            outputs = news_net(titles)
            loss = criterion(outputs, labels)
            avg_validation_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_validation_loss /= validation_batch
        validation_accuracy = correct / total
        writer.add_scalar('validation/loss', avg_validation_loss, epoch)
        writer.add_scalar('validation/accuracy', validation_accuracy, epoch)
        print('Finished epoch [%d/%d], '
              'Average training loss: %.5f (%.5f accuracy), validation loss: %.5f (%.5f accuracy)'
              'Train-Size: %d, Test-Size: %d' %
              (epoch + 1, num_epochs, avg_training_loss, accuracy,
               avg_validation_loss, validation_accuracy, train_size, test_size))

    writer.flush()

    # Save model after training TODO: Careful! This gets big really fast!
    save_model = False

    if save_model:
        os.makedirs("./models", exist_ok=True)
        news_net.to("cpu")
        dummy_input = torch.zeros((input_size))
        traced_model = torch.jit.trace(news_net, dummy_input)
        torch.jit.save(traced_model, "./models/news_net.pt")
        my_model = torch.jit.load("./models/news_net.pt")
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
