import os
import pickle

import gensim.downloader as api
import numpy as np
import spacy
from tqdm import tqdm


class LemmaEmbedding:
    def __init__(self, vocab):
        self.nlp_lemmatizer = spacy.load('en_core_web_sm', disable=['parser'])
        embedding_file = 'embeddings/lemma_embedding.pkl'
        if os.path.isfile(embedding_file):
            with open(embedding_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            self.embeddings = self.calculate_embeddings(vocab)
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
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
        self.embeddings = api.load("glove-wiki-gigaword-100")
        self.embeddings_len = 100

    def embed_word(self, title):
        return self.embeddings.get_mean_vector(title.split())