import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler)
import random

import numpy as np

def data_loader(df, batch_size=50):
    """Convert train and validation sets to torch.Tensors and load them to
    DataLoader.
    """

    x = list([list(a) for a in df.text.values])
    # y = list([list(a) for a in df.label.values])
    # train_inputs, train_labels = tuple(torch.from_numpy(data) for data in [df.text.values, df.label.values])

    train_inputs, train_labels = tuple(torch.tensor(data) for data in [x, df.label.values])


    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=batch_size,drop_last=False)


    return train_dataloader

def bert_Dataloader(tokens,df,batch_size=50):
    data_seq = torch.tensor(tokens['input_ids'])
    data_mask = torch.tensor(tokens['attention_mask'])
    data_y = torch.tensor(df.label.tolist())

    data = TensorDataset(data_seq, data_mask, data_y)
    data_sampler = RandomSampler(data)
    data_loader = DataLoader(data, sampler=data_sampler, batch_size=batch_size, drop_last=False)

    return data_loader

def pad(list, maxlen=200, pad='_pad_'):
    list += [pad] * (maxlen - len(list))
    return list[:maxlen]

def elmo_dataLoader(df, batch_size = 32, maxlen=200, shuffle=False):
    from allennlp.modules.elmo import batch_to_ids
    x_data = [pad(sent,maxlen=maxlen) for sent in df.text.values]
    y_data = df.label.values

    batches = []
    for i in range(0, len(x_data), batch_size):
        start, stop = i, i + batch_size
        x_batch = batch_to_ids(x_data[start:stop])
        y_batch = torch.tensor(torch.from_numpy(np.array(y_data[start:stop])), dtype=torch.long)
        lengths = torch.tensor(torch.from_numpy(np.array([max(len(x), 1) for x in x_data[start:stop]])).int()).view(-1,
                                                                                                                    1)
        batches.append((x_batch, y_batch, lengths))
    if shuffle:
        random.shuffle(batches)
    return batches

