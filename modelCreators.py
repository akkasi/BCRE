from multiViewCNN_Glove import *
from multiView_BioBert import *
from Attention_BILSTM import *
import yaml
config_file = open('configure.yaml')
config = yaml.load(config_file, Loader=yaml.FullLoader)

def mvcc_glove_creator(pretrained_embedding,
                        freeze_embedding,
                        vocab_size,
                        embed_dim,
                        filter_sizes,
                        num_filters,
                        num_classes,
                        dropout):
    model = mvccGlove(
        pretrained_embedding=pretrained_embedding,
        freeze_embedding=freeze_embedding,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        num_classes=num_classes,
        dropout=dropout)
    return model
def attnBilstm_creator(pretrained_embedding = None,
            freeze_embedding = True,
            vocab_size = None,
            embed_dim = 100,
            hidden_size = 100,
            n_classes = 3,
            n_lstm_layers = 2,
            dropout = 0.5,
            device = None):

    model = attn_BiLSTM(pretrained_embedding = pretrained_embedding,
            freeze_embedding = freeze_embedding,
            vocab_size = vocab_size,
            embed_dim = embed_dim,
            hidden_size = hidden_size,
            n_classes = n_classes,
            n_lstm_layers = n_lstm_layers,
            dropout = dropout,
            device = device)
    return model




