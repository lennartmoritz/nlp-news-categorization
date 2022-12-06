---
---
# nlp-news-categorization
Assignment Option Four - News Categorization using PyTorch

Link to our GitHub repository: https://github.com/lennartmoritz/nlp-news-categorization
---
---
# Authors
 * Carlotta Mahncke 
 * Lennart Joshua Moritz
 * Timon Engelke 
 * Christian Schuler
 
--- 
---
# Setup
 * `sudo apt install python3-venv`
 * `python3 -m venv venv`
 * `source venv/bin/activate`
 * Automatic: ./script-caller.bh`
 * Or Manual:
   * `pip install -r requirements.txt`
   * `python -m spacy download "en_core_web_sm"`
 
---
---
# Use
### Multiclass
 * `python news-categorization.py -l b t e m`
### Binary classification
 * `python news-categorization.py -l b t`
### Specify embedding
 * `python news-categorization.py -l b t -e word2vec`
 * Choose from: word, lemma, word2vec, glove

---
---
# Task
Text categorization using PyTorch

 * A typical workflow in PyTorch. 
![images/nlpPyTorchTypicalWorkflow.png](images/nlpPyTorchTypicalWorkflow.png)

 * The task is to classify news articles into one of the following categories: 
   * Business
   * Science and Technology
   * Entertainment
   * Health

# Results

As shown in our presentation, the lemma embedding performed the best, followed by the word embedding.
Pretrained embeddings like word2vec and glove performed worse than the embeddings we trained ourselves.
This is probably because the self-trained embeddings are more specific to our dataset.

For the binary classification tasks, we saw that differentiating between business and science and
technology was the hardest task, while health and entertainment were the easiest.
This is probably because the business and science and technology articles are very similar in their
vocabulary, while the health and entertainment articles are very different.

# Comparison to BERT

You can use the script `bert_classification.py` to fine-tune a pre-trained BERT classifier on our dataset.
The results are slightly better than the results we got with our own models (around 96% accuracy compared to
around 94% accuracy on our model for the multi-class classification task).

---
---
# Dataset 

News Aggregator Dataset
https://www.kaggle.com/datasets/uciml/news-aggregator-dataset

