import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, bertBeaseModel, D = 768, n_C = 3, kernels = [3,4,5], n_kernels = 100, dropout=0.5, freeze = True):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bertBeaseModel)

        # Any kind of network can be replaced with the following archtiture
        self.convs1 = nn.ModuleList([nn.Conv2d(1,n_kernels,(ker,D)) for ker in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=len(kernels) * n_kernels, out_features= n_C)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, input_ids, attention_masks):
        out = self.bert(input_ids = input_ids, attention_mask = attention_masks)

        last_hidden_state_cls = out[0][:,0,:]
        x = last_hidden_state_cls.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d((i,i.size(2)).squeeze(2) for i in x)]
        x = torch.cat(x,1)
        x = self.dropout(x)
        logits = self.fc1(x)
        return logits

