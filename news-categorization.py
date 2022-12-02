#!/usr/bin/env python3
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy


# Split the dataset randomly into train and test
def data_splitter(list_of_datasets):
    # Concatenate the dataframes
    dataframes = []
    for dataframe in list_of_datasets:
        dataframes.append(dataframe)
    dataframes_all = pd.concat(dataframes)
    # Splitting of the data
    train_dataset = dataframes_all.sample(frac=0.8, random_state=0)
    test_dataset = dataframes_all.drop(train_dataset.index)
    return train_dataset, test_dataset

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


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Recommended by error-message

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    # Read dataset
    full_dataset = pd.read_csv('data/uci-news-aggregator.csv')


    # Group data for customized data selection
    grouped_dataset = full_dataset.groupby(full_dataset['CATEGORY'])
    e_dataset = grouped_dataset.get_group('e')
    b_dataset = grouped_dataset.get_group('b')
    t_dataset = grouped_dataset.get_group('t')
    m_dataset = grouped_dataset.get_group('m')

    # Sanity-Check
    print("Number of e: " + str(len(e_dataset)))
    print("Number of b: " + str(len(b_dataset)))
    print("Number of t: " + str(len(t_dataset)))
    print("Number of m: " + str(len(m_dataset)))


    # Duo-Datasets
    eb_train_dataset, eb_test_dataset = data_splitter([e_dataset, b_dataset])
    et_train_dataset, et_test_dataset = data_splitter([e_dataset, t_dataset])
    em_train_dataset, em_test_dataset = data_splitter([e_dataset, m_dataset])

    bt_train_dataset, bt_test_dataset = data_splitter([b_dataset, t_dataset])
    bm_train_dataset, bm_test_dataset = data_splitter([b_dataset, m_dataset])

    tm_train_dataset, tm_test_dataset = data_splitter([t_dataset, m_dataset])

    full_train_dataset, full_test_dataset = data_splitter([e_dataset, b_dataset, t_dataset, m_dataset])

    # TEMP to not interrupt other coder!
    train_dataset = full_train_dataset
    test_dataset = full_test_dataset


    #===DATALOADER===
    data = pd.read_csv("data/uci-news-aggregator.csv")
    embeddings = get_word_embeddings(data)
    categories = ['e', 'b', 't', 'm']

    dataset = CustomNewsDataset(data, embeddings, categories)
    train_size = 100 #int(0.8 * len(dataset))
    test_size = 100 #len(dataset) - train_size
    validate_size = len(dataset) - (train_size + test_size)
    training_dataset, testing_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, validate_size])

    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}

    # dataloader
    train_generator = DataLoader(training_dataset, **params)
    test_generator = DataLoader(testing_dataset, **params)

    # Parameters
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 150
    display_step = 1

    # Network Parameters
    hidden_size = 100  # 1st layer and 2nd layer number of features
    input_size = len(embeddings)  # Words in vocab
    num_classes = 4  # Categories: "e", "b", "t", "m"


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
        for titles, labels in tqdm(train_generator):
            titles, labels = titles.to(device), labels.to(device)

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
    all_predictions = []
    all_labels = []

for test_data, labels in tqdm(test_generator):
    test_data, labels = test_data.to(device), labels.to(device)
    all_labels.append(labels.cpu().detach().numpy())
    outputs = news_net(test_data)
    _, predicted = torch.max(outputs.data, 1)
    all_predictions.append(predicted.cpu().detach().numpy())

    all_predictions = torch.Tensor(np.concatenate(all_predictions))
    all_labels = torch.Tensor(np.concatenate(all_labels))

    accuracy = Accuracy(task='multiclass', num_classes=4)
    acc = accuracy(all_predictions, all_labels)

    print('Accuracy of the model on the test data: %f' % (acc,))
