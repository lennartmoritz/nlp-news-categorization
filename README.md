---
---
# nlp-news-categorization
Assignment Option Four - News Categorization using PyTorch

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
 * sudo apt install python3-venv
 * python3 -m venv venv
 * source venv/bin/activate
 * pip install -r requirements.txt
 * python -m spacy download "en_core_web_sm"
 * python -m spacy download "en_core_web_trf"
 
---
---
# Use
### Multiclass
 * python news-categorization.py -l b t e m
### Binary classification
 * python news-categorization.py -l b t


---
---
# Task
Text categorization using PyTorch

 * A typical workflow in PyTorch. 
![images/nlpPyTorchTypicalWorkflow.png](images/nlpPyTorchTypicalWorkflow.png)


---
---
# Dataset 

News Aggregator Dataset
https://www.kaggle.com/datasets/uciml/news-aggregator-dataset

