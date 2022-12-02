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


# Duo-Datasets
eb_train_dataset, eb_test_dataset = data_splitter([e_dataset, b_dataset])
et_train_dataset, et_test_dataset = data_splitter([e_dataset, t_dataset])
em_train_dataset, em_test_dataset = data_splitter([e_dataset, m_dataset])

bt_train_dataset, bt_test_dataset = data_splitter([b_dataset, t_dataset])
bm_train_dataset, bm_test_dataset = data_splitter([b_dataset, m_dataset])

tm_train_dataset, tm_test_dataset = data_splitter([t_dataset, m_dataset])

# Sanity-Check
print("Number of train eb: " + str(len(eb_train_dataset)) + " Number of test eb: " + str(len(eb_test_dataset)))
print("Number of train et: " + str(len(et_train_dataset)) + " Number of test et: " + str(len(et_test_dataset)))
print("Number of train em: " + str(len(eb_train_dataset)) + " Number of test em: " + str(len(em_test_dataset)))

print("Number of train bt: " + str(len(bt_train_dataset)) + " Number of test bt: " + str(len(bt_test_dataset)))
print("Number of train bm: " + str(len(bm_train_dataset)) + " Number of test bm: " + str(len(bm_test_dataset)))

print("Number of train tm: " + str(len(tm_train_dataset)) + " Number of test tm: " + str(len(tm_test_dataset)))








