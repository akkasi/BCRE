import torch
import torch.nn as nn
import torch.nn.functional as F

class attn_BiLSTM(nn.Module):
    def __init__(self,pretrained_embedding = None,
            freeze_embedding = False,
            vocab_size = None,
            embed_dim = 100,
            hidden_size = 100,
            n_classes = 3,
            n_lstm_layers = 1,
            dropout = 0.5,
            device = None,
            bidirectional = True):
        super(attn_BiLSTM, self).__init__()

        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        self.x = 2 if self.bidirectional else 1

        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)


        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size ,num_layers=self.n_lstm_layers,
                            bidirectional=self.bidirectional, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * self.x ,self.n_classes)

        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )


    def attention(self, rnn_out, state):
        # print('rnn_out, state',rnn_out.size(), state.size())
        merged_state = torch.cat([s for s in state[-2:]], 1)
        # print('merged_state',merged_state.size())
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # print('merged_state', merged_state.size())
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        # print('weights',weights.size())
        weights = torch.nn.functional.softmax(weights.squeeze(2),dim=1).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        # print('weights', weights.size())
        # print('torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)',torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2).size())
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    def forward(self, input_sentences):

        input = self.embedding(input_sentences).float()

        h = torch.zeros((self.n_lstm_layers * self.x , input_sentences.size(0), self.hidden_size )).to(self.device)
        c = torch.zeros((self.n_lstm_layers * self.x , input_sentences.size(0), self.hidden_size  )).to(self.device)

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out, (hidden, cell) = self.lstm(input, (h, c))

        attn_out = self.attention(out, hidden)
        logits = self.fc(attn_out)
        return logits
