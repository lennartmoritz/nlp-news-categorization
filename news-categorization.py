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
from sklearn.metrics import confusion_matrix
import sys
import os
import argparse
import spacy


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


class LemmaEmbedding:
    def __init__(self, vocab):
        self.nlp_lemmatizer = spacy.load('en_core_web_sm', disable=['parser'])
        self.embeddings = self.calculate_embeddings(vocab)
        self.embeddings_len = len(self.embeddings)

    # Generate the word embeddings for the selected dataset
    # The basic idea of word embedding is words that occur in similar context tend to be closer to each other in vector space.
    def calculate_embeddings(self, dataset):
        """
        Returns:
        word_embeddings:    A dict that maps word names as keys to an automatically generated word_id
        """
        # Getting all the vocabularies and indexing to a unique position
        vocab = set()
        # Lemmatizing words from titles
        for text in tqdm(dataset['TITLE']):
            doc = self.nlp_lemmatizer(text)
            for word in doc:
                vocab.add(word.lemma_)

        # Build word-embeddings vector for the entire data
        word_embeddings = {}
        for i, word in enumerate(list(vocab)):
            word_embeddings[word] = i

        return word_embeddings

    def embed_word(self, title):
        title = self.nlp_lemmatizer(title)

        embedded_title = np.zeros(len(self.embeddings), dtype=np.float64)

        for token in title:
            embedded_title[self.embeddings[token.lemma_]] += 1

        return embedded_title


class WordEmbedding:
    def __init__(self, vocab):
        self.embeddings = self.calculate_embeddings(vocab)
        self.embeddings_len = len(self.embeddings)

    # Generate the word embeddings for the selected dataset
    # The basic idea of word embedding is words that occur in similar context tend to be closer to each other in vector space.
    def calculate_embeddings(self, dataset):
        """
        Returns:
        word_embeddings:    A dict that maps word names as keys to an automatically generated word_id
        """
        # Getting all the vocabularies and indexing to a unique position
        vocab = set()
        # Lemmatizing words from titles
        for text in tqdm(dataset['TITLE']):
            for word in text.split(' '):
                vocab.add(word)

        # Build word-embeddings vector for the entire data
        word_embeddings = {}
        for i, word in enumerate(list(vocab)):
            word_embeddings[word] = i

        return word_embeddings

    def embed_word(self, title):
        embedded_title = np.zeros(len(self.embeddings), dtype=np.float64)

        for word in title.split(' '):
            embedded_title[self.embeddings[word]] += 1

        return embedded_title


class PretrainedEmbedding:
    def __init__(self, vocab):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.nlp = spacy.load("en_core_web_trf")
        self.embeddings_len = 768

    def embed_word(self, title):
        doc = self.nlp(title)
        return doc._.trf_data.tensors[-1][0]

class CustomNewsDataset(Dataset):
    def __init__(self, data, embedding, categories):
        self.labels = data['CATEGORY']
        self.titles = data['TITLE']
        self.embedding = embedding
        self.categories = categories

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles.iloc[idx]
        embedded_title = self.embedding.embed_word(title)

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

    # Accepting arguments to select datasets (classes) for classification task
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)
    input_parser.add_argument('-e', '--embedding', default='lemma', choices=['lemma', 'word', 'pretrained'])
    # Used like:
    # python arg.py -l b t m e      => multiclass 
    # python arg.py -l b t          => binary

    args = input_parser.parse_args()
    
    list_of_classes = list()
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
    data_all = pd.read_csv("data/uci-news-rand-reduced.csv")
    
    # Preparing datasets based on input parameters (list of classes)
    list_of_datasets = []   # Splittet based on class_key = {e, b, t, m}
    for class_key in list_of_classes:
        data_subset = data_splitter(data_all, 'CATEGORY', class_key)
        list_of_datasets.append(data_subset)
    
    # Merge dataset to contain all required classes
    data = data_merger(list_of_datasets)

    unique_categories = len(data['CATEGORY'].unique())



    # Initialize spacy 'en_core_web_sm' model, keeping only tagger component needed for lemmatization
    #nlp_lemmatizer = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
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
    
    #TODO: check that the datasets are balanced, i.e. all categories must appear
    training_dataset, testing_dataset, validating_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, validate_size])

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

    # Save model after training TODO: Careful! This gets big really fast!
    save_model=False
    
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
    labels_occurences = Counter(all_labels.tolist())
    print('Occurences of labels/classes: %s' % (labels_occurences))

    ## Specificity
    resulting_confusion_matrix = confusion_matrix(all_labels, all_predictions)
    FP = resulting_confusion_matrix.sum(axis=0) - np.diag(resulting_confusion_matrix)  
    FN = resulting_confusion_matrix.sum(axis=1) - np.diag(resulting_confusion_matrix)
    TP = np.diag(resulting_confusion_matrix)
    TN = resulting_confusion_matrix.sum() - (FP + FN + TP)


    # === All types of evaluations ===
    
    # Evaluation on scope of class level
    TPR = TP/(TP+FN)    # Sensitivity, hit rate, recall, or true positive rate
    TNR = TN/(TN+FP)    # Specificity or true negative rate
    PPV = TP/(TP+FP)    # Precision or positive predictive value
    NPV = TN/(TN+FN)    # Negative predictive value
    FPR = FP/(FP+TN)    # Fall out or false positive rate
    FNR = FN/(TP+FN)    # False negative rate
    FDR = FP/(TP+FP)    # False discovery rate
    ACC = (TP+TN)/(TP+FP+FN+TN)    # Overall accuracy
    F1S = 2*(PPV*TPR)/(PPV+TPR)    # F1-Score
    evaluations = [FP, FN, TP, TN, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, F1S]
    
    # Evaluation on scope above of class level
    FPa = sum(FP)    # Summing the values for all classes together
    FNa = sum(FN)
    TPa = sum(TP)
    TNa = sum(TN)
    TPRa = TPa/(TPa+FNa)    # Sensitivity, hit rate, recall, or true positive rate
    TNRa = TNa/(TNa+FPa)    # Specificity or true negative rate
    PPVa = TPa/(TPa+FPa)    # Precision or positive predictive value
    NPVa = TNa/(TNa+FNa)    # Negative predictive value
    FPRa = FPa/(FPa+TNa)    # Fall out or false positive rate
    FNRa = FNa/(TPa+FNa)    # False negative rate
    FDRa = FPa/(TPa+FPa)    # False discovery rate
    ACCa = (TPa+TNa)/(TPa+FPa+FNa+TNa)    # Overall accuracy
    F1Sa = 2*(PPVa*TPRa)/(PPVa+TPRa)      # F1-Score
    evaluationsa = [FPa, FNa, TPa, TNa, TPRa, TNRa, PPVa, NPVa, FPRa, FNRa, FDRa, ACCa, F1Sa]

    # Save evaluations to file
    save_eval=False

    if save_eval:
        os.makedirs("./evaluations", exist_ok=True)
        current_classes_as_string = ""
        for class_key in list_of_classes:
            current_classes_as_string = current_classes_as_string + class_key
        current_filename = "evaluation-" + current_classes_as_string + "-" + str(train_size) + "-" + str(num_epochs)
        current_file = "./evaluations/" + current_filename + ".txt"
        with open(current_file, 'w') as file:
            file.write(str(labels_occurences)+"\n")
            file.write("\n")            
            for element in evaluations:
                file.write(str(element)+"\n")
            file.write("\n")
            for element in evaluationsa:
                file.write(str(element)+"\n")

    # Display evaluations on console
    display_eval = True

    if display_eval:
        # Evaluation on scope of class level
        print("Evaluation on scope of class level")
        print(f"FP: {FP}     FN: {FN}     TP: {TP}     TN: {TN}")

        print(f"{'Sensitivity, hit rate, recall, or true positive rate:':<55}{TPR}")
        print(f"{'Specificity or true negative rate:':<55}{TNR}")
        print(f"{'Precision or positive predictive value:':<55}{PPV}")
        print(f"{'Negative predictive value:':<55}{NPV}")
        print(f"{'Fall out or false positive rate:':<55}{FPR}")
        print(f"{'False negative rate:':<55}{FNR}")
        print(f"{'False discovery rate:':<55}{FDR}")
        print(f"{'Overall accuracy:':<55}{ACC}") 
        print(f"{'F1-Score:':<55}{F1S}") 
        
        print("")
        
        # Evaluation on scope above of class level
        print("Evaluation on scope above of class level")
        print(f"FP: {FPa}     FN: {FNa}     TP: {TPa}     TN: {TNa}")

        print(f"{'Sensitivity, hit rate, recall, or true positive rate:':<55}{TPRa:>12}")
        print(f"{'Specificity or true negative rate:':<55}{TNRa:>12}")
        print(f"{'Precision or positive predictive value:':<55}{PPVa:>12}")
        print(f"{'Negative predictive value:':<55}{NPVa:>12}")
        print(f"{'Fall out or false positive rate:':<55}{FPRa:>12}")
        print(f"{'False negative rate:':<55}{FNRa:>12}")
        print(f"{'False discovery rate:':<55}{FDRa:>12}")
        print(f"{'Overall accuracy:':<55}{ACCa:>12}") 
        print(f"{'F1-Score:':<55}{F1Sa:>12}") 
