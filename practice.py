#library imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
import en_core_web_sm
# import jovian
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error

#input
x = torch.tensor([[1,2, 12,34, 56,78, 90,80],
                 [12,45, 99,67, 6,23, 77,82],
                 [3,24, 6,99, 12,56, 21,22]])
print(x.size())
model1 = nn.Embedding(100, 7, padding_idx=0)
model2 = nn.LSTM(input_size=7, hidden_size=3, num_layers=2,bidirectional=False, batch_first=True)
model3 = nn.Sequential(nn.Embedding(100, 7, padding_idx=0),
                        nn.LSTM(input_size=7, hidden_size=3, num_layers=1, batch_first=True))
out1 = model1(x)
out2 = model2(out1)
out, (ht, ct) = model3(x)
# print(out)

# out, (ht, ct) = model2(out1)
# print(ht)

#loading the data
reviews = pd.read_csv("/home/akkasi/pythonProject/‌BERT_CNN_TextClassification/data/Womens Clothing E-Commerce Reviews.csv")
# print(reviews.shape)
# print(reviews.head())

reviews['Title'] = reviews['Title'].fillna('')
reviews['Review Text'] = reviews['Review Text'].fillna('')
reviews['review'] = reviews['Title'] + ' ' + reviews['Review Text']

#keeping only relevant columns and calculating sentence lengths
reviews = reviews[['review', 'Rating']]
reviews.columns = ['review', 'rating']
reviews['review_length'] = reviews['review'].apply(lambda x: len(x.split()))
# reviews.head()

#changing ratings to 0-numbering
zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
reviews['rating'] = reviews['rating'].apply(lambda x: zero_numbering[x])

reviews = reviews[:100]

#tokenization
# tok = spacy.load('en')
tok = en_core_web_sm.load()
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

#count number of occurences of each word
counts = Counter()
for index, row in reviews.iterrows():
    counts.update(tokenize(row['review']))

#deleting infrequent words
# print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
# print("num_words after:",len(counts.keys()))

#creating vocabulary
vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)

def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length

reviews['encoded'] = reviews['review'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))
# print(reviews.head())
#
# #check how balanced the dataset is
# print(Counter(reviews['rating']))
#
X = list(reviews['encoded'])
y = list(reviews['rating'])
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# print(X_train[1],torch.from_numpy(X_train[1][0].astype(np.int32)), y_train[1], X_train[1][1])
# #Pytorch Dataset
#
class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]
train_ds = ReviewsDataset(X_train, y_train)
valid_ds = ReviewsDataset(X_valid, y_valid)
#
#
def train_model(model,train_dl, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        print(pred)
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total
#
batch_size = 10
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)

#LSTM with fixed length input
class LSTM_fixed_len(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True,batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


model_fixed =  LSTM_fixed_len(vocab_size, 50, 50)
train_model(model_fixed,train_dl=train_dl, epochs=2, lr=0.01)
#
# #LSTM with variable length input
#
# class LSTM_variable_input(torch.nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.dropout = nn.Dropout(0.3)
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, 5)
#
#     def forward(self, x, s):
#         x = self.embeddings(x)
#         x = self.dropout(x)
#         x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
#         out_pack, (ht, ct) = self.lstm(x_pack)
#         out = self.linear(ht[-1])
#         return out
# model = LSTM_variable_input(vocab_size, 50, 50)
# train_model(model,train_dl=train_dl,  epochs=30, lr=0.1)
#
# #LSTM with pretrained Glove word embeddings
# def load_glove_vectors(glove_file="./data/glove.6B/glove.6B.50d.txt"):
#     """Load the glove word vectors"""
#     word_vectors = {}
#     with open(glove_file) as f:
#         for line in f:
#             split = line.split()
#             word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
#     return word_vectors
#
# def get_emb_matrix(word_vecs, word_counts, emb_size = 50):
#     """ Creates embedding matrix from word vectors"""
#     vocab_size = len(word_counts) + 2
#     vocab_to_idx = {}
#     vocab = ["", "UNK"]
#     W = np.zeros((vocab_size, emb_size), dtype="float32")
#     W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
#     W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words
#     vocab_to_idx["UNK"] = 1
#     i = 2
#     for word in word_counts:
#         if word in word_vecs:
#             W[i] = word_vecs[word]
#         else:
#             W[i] = np.random.uniform(-0.25,0.25, emb_size)
#         vocab_to_idx[word] = i
#         vocab.append(word)
#         i += 1
#     return W, np.array(vocab), vocab_to_idx
# word_vecs = load_glove_vectors()
# pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, counts)
#
#
# class LSTM_glove_vecs(torch.nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights):
#         super().__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
#         self.embeddings.weight.requires_grad = False  ## freeze embeddings
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, 5)
#         self.dropout = nn.Dropout(0.2)
#
#     def forward(self, x, l):
#         x = self.embeddings(x)
#         x = self.dropout(x)
#         lstm_out, (ht, ct) = self.lstm(x)
#         return self.linear(ht[-1])
# model = LSTM_glove_vecs(vocab_size, 50, 50, pretrained_weights)
# train_model(model, train_dl=train_dl, epochs=30, lr=0.1)
#
# #Predicting ratings using regression instead of classification
# def train_model_regr(model, epochs=10, lr=0.001):
#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = torch.optim.Adam(parameters, lr=lr)
#     for i in range(epochs):
#         model.train()
#         sum_loss = 0.0
#         total = 0
#         for x, y, l in train_dl:
#             x = x.long()
#             y = y.float()
#             y_pred = model(x, l)
#             optimizer.zero_grad()
#             loss = F.mse_loss(y_pred, y.unsqueeze(-1))
#             loss.backward()
#             optimizer.step()
#             sum_loss += loss.item()*y.shape[0]
#             total += y.shape[0]
#         val_loss = validation_metrics_regr(model, val_dl)
#         if i % 5 == 1:
#             print("train mse %.3f val rmse %.3f" % (sum_loss/total, val_loss))
#
# def validation_metrics_regr (model, valid_dl):
#     model.eval()
#     correct = 0
#     total = 0
#     sum_loss = 0.0
#     for x, y, l in valid_dl:
#         x = x.long()
#         y = y.float()
#         y_hat = model(x, l)
#         loss = np.sqrt(F.mse_loss(y_hat, y.unsqueeze(-1)).item())
#         total += y.shape[0]
#         sum_loss += loss.item()*y.shape[0]
#     return sum_loss/total
#
#
# class LSTM_regr(torch.nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim):
#         super().__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, 1)
#         self.dropout = nn.Dropout(0.2)
#
#     def forward(self, x, l):
#         x = self.embeddings(x)
#         x = self.dropout(x)
#         lstm_out, (ht, ct) = self.lstm(x)
#         return self.linear(ht[-1])
# model =  LSTM_regr(vocab_size, 50, 50)
# train_model_regr(model, train_dl=train_dl, epochs=30, lr=0.05)
