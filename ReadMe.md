# Sentiment Analysis
Sentiment analysis with LSTM using pytorch.

### Dataset
Trained using Twitter Airline Sentiment data from [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment).

Download the data and copy in the dataset folder. 

### Setup
Download gloVe vector from [here](http://nlp.stanford.edu/data/glove.6B.zip)

Extract and copy contents to "data" folder.

Execute `python build.py` to prepare the vocabulary and compressed word vectors.

### Training
```python train.py``` for training.

### Evaluation
```python evaluate.py``` for command line prompt for evaluation.