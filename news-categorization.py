#!/usr/bin/env python3
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device:', device)

full_dataset = pd.read_csv('data/uci-news-aggregator.csv')

# Split the dataset randomly into train and test
train_dataset = full_dataset.sample(frac=0.8, random_state=0)
test_dataset = full_dataset.drop(train_dataset.index)

# Getting all the vocabularies and indexing to a unique position
vocab = Counter()
# Indexing words from the training data
for text in train_dataset['TITLE']:
    for word in text.split(' '):
        vocab[word.lower()] += 1

# Indexing words from the test data
for text in test_dataset['TITLE']:
    for word in text.split(' '):
        vocab[word.lower()] += 1

total_words = len(vocab)


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


word2index = get_word_2_index(vocab)


def get_batch(df, i, batch_size):
    batches = []
    results = []
    # Split into different batchs, get the next batch
    titles = df['TITLE'].iloc[i * batch_size:i * batch_size + batch_size]
    # get the targets
    categories = df['CATEGORY'].iloc[i * batch_size:i * batch_size + batch_size]
    for title in titles:
        layer = np.zeros(total_words, dtype=float)

        for word in title.split(' '):
            layer[word2index[word.lower()]] += 1
        batches.append(layer)

    # We have 4 categories
    for category in categories:
        index_y = -1
        if category == 'e':
            index_y = 0
        elif category == 'b':
            index_y = 1
        elif category == 't':
            index_y = 2
        elif category == 'm':
            index_y = 3
        results.append(index_y)

    # the training and the targets
    return np.array(batches), np.array(results)


# Parameters
learning_rate = 0.01
num_epochs = 10
batch_size = 150
display_step = 1

# Network Parameters
hidden_size = 100  # 1st layer and 2nd layer number of features
input_size = total_words  # Words in vocab
num_classes = 4  # Categories: "e", "b", "t", "m"


# define the network
class NewsClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

    # accept input and return an output
    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out


news_net = NewsClassifier(input_size, hidden_size, num_classes)
news_net = news_net.to(device)
news_net.train()
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # This includes the Softmax loss function
optimizer = torch.optim.Adam(news_net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    # shuffle the data
    train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
    # determine the number of min-batches based on the batch size and size of training data
    total_batch = int(len(train_dataset) / batch_size)
    avg_loss = 0
    # Loop over all batches
    for i in tqdm(range(total_batch)):
        batch_x, batch_y = get_batch(train_dataset, i, batch_size)
        titles = torch.FloatTensor(batch_x)
        titles = titles.to(device)
        labels = torch.LongTensor(batch_y)
        labels = labels.to(device)
        # print("articles",articles)
        # print(batch_x, labels)
        # print("size labels",labels.size())

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = news_net(titles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

    avg_loss = avg_loss / total_batch
    print('Finished epoch [%d/%d], Average loss: %.4f' % (epoch + 1, num_epochs, avg_loss))

# Calculate Accuracy
news_net.eval()
correct = 0
total = 0
for i in range(len(test_dataset)):
    test_x, test_y = get_batch(test_dataset, i, 1)
    titles = torch.FloatTensor(test_x)
    titles = titles.to(device)
    outputs = news_net(titles).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += 1
    correct += 1 if predicted == test_y else 0

print('Accuracy of the model on the test data: %d %%' % (100 * correct / total))
