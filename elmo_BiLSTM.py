import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.commands.elmo import ElmoEmbedder
class ELMo_BiLSTM(nn.Module):
    def __init__(self,
            hidden_size = 100,
            n_classes = 3,
            n_lstm_layers = 1,
            dropout = 0.5,
            device = None,
            bidirectional = True):
        super(ELMo_BiLSTM, self).__init__()
        self.elmo =  ElmoEmbedder(
                options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
                weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
                                    )
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.n_classes =n_classes
        self.device = device
        self.bidirectional = bidirectional
        self.x = 2 if self.bidirectional else 1
        self.lstm = nn.LSTM(1024, self.hidden_size, num_layers=self.n_lstm_layers,
                            bidirectional=self.bidirectional, batch_first=True)
        self.fc = nn.Linear(self.hidden_size , self.n_classes)

        self.activation_func = nn.ReLU6()
        self.dropout_l = nn.Dropout(dropout)

    def forward(self, sentences):

        elmo_out = self.elmo(sentences)
        x = elmo_out['elmo_representations'][0]
        # x = x.transpose(1,2)
        h = torch.zeros((self.n_lstm_layers * self.x, sentences.size(0), self.hidden_size)).to(self.device)
        c = torch.zeros((self.n_lstm_layers * self.x, sentences.size(0), self.hidden_size)).to(self.device)

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out, (hidden, cell) = self.lstm(x, (h, c))


        x = self.activation_func(hidden[-1])
        x = self.p1(x)
        x = self.dropout_l(x)
        y = self.fc(x)
        return y
