import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
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
                                  batch_size=batch_size,drop_last=True)


    return train_dataloader

def bert_Dataloader(df,batch_size=50):
    df.input_ids = torch.tensor(df.input_ids.values)
    df.input_masks = torch.tensor(df.input_masks.values)
    df.labels = torch.tensor(df.label.values)
    data = TensorDataset(df.input_ids,df.input_masks,df.label)
    data_sampler = RandomSampler(data)
    data_loader = DataLoader(data, sampler=data_sampler,batch_size=batch_size)
    return data_loader

    


