# Sentiment Analysis
Sentiment analysis with LSTM using pytorch.

### Dataset
- Trained using Twitter Airline Sentiment data from [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment).

- Download the data and copy in the dataset folder. 

### Setup
1. Download gloVe vector from [here](http://nlp.stanford.edu/data/glove.6B.zip)

2. Extract and copy contents to "data" folder.

3. Execute `python build.py` to prepare the vocabulary and compressed word vectors.

### Training
- ```python train.py``` for training.

### Evaluation
- ```python evaluate.py``` for command line prompt for evaluation.