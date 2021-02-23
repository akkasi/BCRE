import torch
import torch.nn as nn

class elmoBilstm(nn.Module):
    def __init__(self,
            embed_dim = 1024,
            hidden_size = 100,
            n_classes = 3,
            n_lstm_layers = 1,
            dropout = 0.5,
            device = None,
            bidirectional = True,
            elmo = None
            ):

        super(elmoBilstm, self).__init__()
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.device = device
        self.elmo = elmo
        self.dropout = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        self.x = 2 if self.bidirectional else 1
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size ,num_layers=self.n_lstm_layers,
                            bidirectional=self.bidirectional, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * self.x ,self.n_classes)

    def forward(self, input_sentences):
        embeddings_allennlp = self.elmo(input_sentences)
        input = embeddings_allennlp['elmo_representations'][0]

        h = torch.zeros((self.n_lstm_layers * self.x , input_sentences.size(0), self.hidden_size )).to(self.device)
        c = torch.zeros((self.n_lstm_layers * self.x , input_sentences.size(0), self.hidden_size  )).to(self.device)

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out, (hidden, cell) = self.lstm(input, (h, c))

        logits = self.fc(out[:, -1])
        return logits