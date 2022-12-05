import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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


def check_dataset_balance(this_dataset, intended_num_of_classes) -> bool:
    """
    Returns True if intended number of unique classes are present in this dataset
    """
    print("Checking that all unique classes are represented...")
    duplicate_removal_set = set()
    for item in tqdm(this_dataset):
        duplicate_removal_set.add(item[1])
    return len(duplicate_removal_set) == intended_num_of_classes


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
