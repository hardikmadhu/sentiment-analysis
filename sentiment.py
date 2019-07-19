# -*- coding: utf-8 -*-
"""sentiment.py

Author: Hardik Madhu (hardikmadhu474@gmail.com)
"""

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pandas
import numpy as np
from tqdm import tqdm
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import nltk
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

UNK = "$UNK$"
PAD = "$PAD$"
embedding_dim = 300
BATCH_SIZE = 32
# MAX_SIZE = 32
layer_dim = 256

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

glove_vocab = None

def create_embedding_vector(embedding_vector):
    embedding_vector = torch.tensor(embedding_vector)
    num_embeddings, embedding_dim = embedding_vector.size()

    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': embedding_vector})
    emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


def load_glove():
    glove_vocab = set()

    glove_file = open("data/glove.6B.300d.txt")

    for line in glove_file.readlines():
        glove_vocab.add(line.split()[0])

    glove_vocab = list(glove_vocab)

    return glove_vocab


def get_compressed_glove(vocab):
    print("Building Embedding Vector")
    embedding_vector = np.zeros((len(vocab), embedding_dim))

    glove_file = open("data/glove.6B.300d.txt")

    for line in glove_file.readlines():
        words = line.split()

        if words[0] in vocab:
            embedding_vector[vocab.index(words[0])] = np.asarray([float(x) for x in words[1:]])

    np.savez_compressed("data/glove_embedding.txt", embeddings=embedding_vector)

    print("Building Embedding Vector Done")
    return embedding_vector


def get_trimmed_glove_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]


def tokenizer(text_line):
    tokens = word_tokenize(text_line)
    filtered = [w.lower() for w in tokens if not w.lower() in stop_words]

    words = [word if word in glove_vocab else UNK for word in filtered]

    return words

  
def evaluate_prediction(y_true_list, y_pred_list, process_tag):
    print("")
    print(process_tag, "Precision Macro", precision_score(y_true_list, y_pred_list, average='macro'))
    print(process_tag, "Recall Macro", recall_score(y_true_list, y_pred_list, average='macro'))
    print(process_tag, "F1 Macro", f1_score(y_true_list, y_pred_list, average='macro'))
    print("")

    f1 = f1_score(y_true_list, y_pred_list, average='macro')
    
    return f1


def build_vocab(tweets_text):
    print("Building Vocabulary")
    vocab = set()

    pbar = tqdm(total=len(tweets_text))

    for complaint in tweets_text:
        words = tokenizer(complaint)

        vocab.update(words)

        pbar.update(1)

    vocab.add(PAD)
    pbar.close()

    vocab = list(vocab)
    print(len(vocab))
    vocab = sorted(vocab)
    print("Building Vocabulary Done")
    return vocab


def read_data():
    tweets_df = pandas.read_csv("dataset/twitter-airline-sentiment/Tweets.csv")
    tweets_df = tweets_df[0:100]
    
    tweets_df.loc[tweets_df['airline_sentiment'] == "negative", 'airline_sentiment'] = 1
    tweets_df.loc[tweets_df['airline_sentiment'] == "positive", 'airline_sentiment'] = 0
    tweets_df.loc[tweets_df['airline_sentiment'] == "neutral", 'airline_sentiment'] = 0
    
    tweets_text = tweets_df['text'].values

    sentiment = tweets_df['airline_sentiment'].values

    return tweets_text, sentiment


def write_vocab(filename, vocab):

    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    l = list()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            l.append(word)

    return l


def build_data(tweets_text):
    vocab = build_vocab(tweets_text)
    write_vocab("data/word_vocab.txt", vocab)

    embedding_vector = get_compressed_glove(vocab)

    return vocab, embedding_vector


class SequenceModel(nn.Module):
    def __init__(self, weights_matrix, hidden_size):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size

        self.word_embeddings, num_embeddings, embedding_dim = create_embedding_vector(weights_matrix)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)
        
        self.lstm_2 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True)
        
        self.dense = nn.Linear(2 * hidden_size, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Sigmoid()
        
        self.drop_layer = nn.Dropout(p=0.5)


    def forward(self, sequence, training=False):
        embeds = self.word_embeddings(sequence)
        if training:
          embeds = self.drop_layer(embeds)

        lstm_out, _ = self.lstm(embeds) 
        if training:
          lstm_out = self.drop_layer(lstm_out)        

        lstm_out_2, _ = self.lstm_2(lstm_out)
        if training:
          lstm_out_2 = self.drop_layer(lstm_out_2)

        max_pool = F.adaptive_max_pool1d(lstm_out_2.permute(1, 2, 0), 1).view(sequence.size(1), -1)

        dense_op = self.dense(max_pool)
        if training:
            dense_op = self.drop_layer(dense_op)

        tag_space = self.dense_2(dense_op)

        tag_scores = self.softmax(tag_space)

        tag_scores = torch.squeeze(tag_scores)

        return tag_scores


def minibatches(dataset, vocab, minibatch_size=BATCH_SIZE):
    sentences, tags = [list(d) for d in list(zip(*dataset))]
    
    num_batches = int(len(sentences) / minibatch_size)

    i = 0
    for i in range(num_batches):
        batch_idxs = []
        batch_lens = []

        batch_sentence = sentences[i*minibatch_size:(i+1)*minibatch_size]
        batch_tags = tags[i*minibatch_size:(i+1)*minibatch_size]

        for sentence in batch_sentence:
            word_idxs = sentence[:]
            batch_lens.append(len(word_idxs))
            batch_idxs.append(word_idxs)
        
        max_len = max(batch_lens)
        word_idxs = [torch.tensor([vocab.index(PAD)]*(max_len - len(idxs))+idxs) for idxs in batch_idxs]
        word_idxs = pad_sequence(word_idxs, padding_value=vocab.index(PAD))

        batch_tags = torch.tensor(batch_tags, dtype=torch.float32)

        yield word_idxs, batch_tags
    
    if i == 0:
      i = -1
    
    batch_idxs = []
    batch_lens = []

    batch_sentence = sentences[(i+1)*minibatch_size:]
    batch_tags = tags[(i+1)*minibatch_size:]

    if len(batch_tags) > 0:
      for sentence in batch_sentence:
          word_idxs = sentence[:]
          batch_lens.append(len(word_idxs))
          batch_idxs.append(word_idxs)

      max_len = max(batch_lens)
      word_idxs = [torch.tensor([vocab.index(PAD)]*(max_len - len(idxs))+idxs) for idxs in batch_idxs]
      word_idxs = pad_sequence(word_idxs, padding_value=vocab.index(PAD))

      batch_tags = torch.tensor(batch_tags, dtype=torch.float32)

      yield word_idxs, batch_tags

        
def run_epoch(dataset, sequence_model, optimizer, loss_function, vocab):
    sentences, _ = [list(d) for d in list(zip(*dataset))]

    pbar = tqdm(total=int(len(sentences)/BATCH_SIZE)+1)
    
    y_true_list = []
    y_pred_list = []

    for sentence, tags in minibatches(dataset, vocab):
        optimizer.zero_grad()
        sentence = sentence.to(device)
        tags = tags.to(device)
        tag_scores = sequence_model(sentence, training=True)

        loss = loss_function(tag_scores, tags)
        loss.backward()
        optimizer.step()
        pbar.update(1)
        
        y_true = tags.to('cpu').data.numpy()
        y_pred = [1 if y_op > 0.5 else 0 for y_op in tag_scores.to('cpu').data.numpy()]
        
        y_true_list.extend(list(y_true))
        y_pred_list.extend(list(y_pred))
        
    pbar.close()
    evaluate_prediction(y_true_list, y_pred_list, "Train")

    
def evaluate(dataset, sequence_model, vocab, process_tag):
    _, y_true_list = [list(d) for d in list(zip(*dataset))]
    
    y_pred_list = []
    y_logs_list = []
    
    with torch.no_grad():
      
      pbar = tqdm(total=int(len(y_true_list)/BATCH_SIZE)+1)

      for sentence, _ in minibatches(dataset, vocab):
        sentence = sentence.to(device)
        tag_scores = sequence_model(sentence, training=False)

        y_pred = [1 if y_op > 0.5 else 0 for y_op in tag_scores.to('cpu').data.numpy()]
        y_pred_list.extend(y_pred)
        
        y_logs_list.append(tag_scores.to('cpu').data.numpy())
        
        pbar.update(1)
      
      pbar.close()

      f1 = evaluate_prediction(y_true_list, y_pred_list, process_tag)

    return f1, y_pred_list, y_logs_list
      
    
def train(train_data, val_data, vocab, embedding_vector):
    sequence_model = SequenceModel(embedding_vector, layer_dim)

    sequence_model.to(device)

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(sequence_model.parameters(), lr=0.003)

    prev_f1 = 0
    
    for epoch in range(20):
        print("\nEpoch", epoch)
        print("\nTraining")
        run_epoch(train_data, sequence_model, optimizer, loss_function, vocab)
        print("\nEvaluating")
        f1, _, _ = evaluate(val_data, sequence_model, vocab, "Eval")
        
        if f1 > prev_f1:
          print("Better F1")
          print("Prev", prev_f1, "New", f1)
          prev_f1 = f1
          torch.save(sequence_model.state_dict(), "model/sequence_model")

          
def test(dataset, dataset_text, vocab, embedding_vector):
    print("Testing")
    with torch.no_grad():
      sequence_model = SequenceModel(embedding_vector, layer_dim)
      sequence_model.load_state_dict(torch.load("model/sequence_model", map_location=device))
      sequence_model.to(device)

      f1, y_pred, y_logs = evaluate(dataset, sequence_model, vocab, "Test")

      _, y_true = [list(d) for d in list(zip(*dataset))]

      ans_dict = {"sentences": dataset_text, "y_true": y_true, "y_pred": y_pred}

      df = pandas.DataFrame(ans_dict)

      df.to_csv("data/answer.txt", sep="\t")

def get_text_idxs(tweets_text, vocab):
    print("Building idxs")

    its = int(len(tweets_text)/BATCH_SIZE)

    pbar = tqdm(total=its)
    tweets_idxs = []
    for i in range(its):
        tweets_idxs.extend([[vocab.index(word) for word in tokenizer(sentence)] for sentence in tweets_text[i*BATCH_SIZE:(i+1)*BATCH_SIZE]])
        pbar.update(1)
    pbar.close()

    return tweets_idxs

def load_data():
    vocab = load_vocab("data/word_vocab.txt")
    embedding_vector = get_trimmed_glove_vectors("data/glove_embedding.txt.npz")

    return vocab, embedding_vector
  

def interactive_eval(vocab, embedding_vector):
    sequence_model = SequenceModel(embedding_vector, layer_dim)
    sequence_model.load_state_dict(torch.load("model/sequence_model", map_location=device))  # Choose whatever GPU device number you want
    sequence_model.to(device)

    print(""" This is an interactive mode. To exit, enter 'exit'. 
              You can enter a sentence like 
              input> I love Paris""")
    
    with torch.no_grad():
        while True:
          sentence = input("input> ")

          words_raw = sentence.strip().split(" ")

          if words_raw == ["exit"]:
              break

          word_idxs = torch.tensor([vocab.index(word) if word in vocab else vocab.index(UNK) for word in tokenizer(sentence)])

          word_idxs = pad_sequence([word_idxs])

          word_idxs = word_idxs.to(device)
          ans = sequence_model(word_idxs, training=False)
          ans = ans.to("cpu").data.numpy()
          if ans <= 0.5:
            ans = "Not Negative with {0:.2f}% confidence".format((1 - ans)*100)
          elif ans > 0.5:
            ans = "Negative with {0:.2f}% confidence".format(ans*100)
            
          print("Input sentence is", ans)


def main(building=False, training=False):
    global glove_vocab
    glove_vocab = load_glove()

    if building or training:
        tweets_text, tweets_sentiment = read_data()
      
    if building:  
        vocab, embedding_vector = build_data(tweets_text)
    else:
        vocab, embedding_vector = load_data()
    
    if training:
        tweets_idxs = get_text_idxs(tweets_text, vocab)

        print(tweets_text[0])

        total_len = len(tweets_text)
        train_len = int(0.8*total_len)
        valid_len = int(0.1*total_len)
        test_len = total_len - (train_len+valid_len)

        dataset = [list(abc) for abc in zip(tweets_idxs, tweets_sentiment)]
        train_data = dataset[:train_len]
        val_data = dataset[train_len:train_len+valid_len]
        test_data = dataset[train_len+valid_len:]

        train_text = tweets_text[:train_len]
        val_text = tweets_text[train_len:train_len+valid_len]
        test_text = tweets_text[train_len+valid_len:]

        train(train_data, val_data, vocab, embedding_vector)
        test(val_data, val_text, vocab, embedding_vector)
        test(train_data, train_text, vocab, embedding_vector)
        
    else:
        if not building:
            interactive_eval(vocab, embedding_vector)

if __name__ == "__main__":
    main(building=True, training=True)
