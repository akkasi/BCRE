from multiViewCNN_g import *
from multiView_BioBert import *
from dataLoader import bert_Dataloader
from elmo_BiLSTM import *
from transformers import AdamW,get_linear_schedule_with_warmup
import yaml
config_file = open('configure.yaml')
config = yaml.load(config_file, Loader=yaml.FullLoader)

def initialize_model(bertClassifier, device,lenght_train_loader, epochs=4):

    bertClassifier.to(device)

    optimizer = AdamW(bertClassifier.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )
    total_steps = lenght_train_loader * epochs
    #
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return optimizer, scheduler

def BertModel(baseModel,n_classes,kernels,n_kernels,dropout,freeze, device, lenght_train_loader):
    bertClassifier = BertClassifier(baseModel, D=768, n_C=n_classes,
                                    kernels=kernels, n_kernels=n_kernels,
                                    dropout=dropout, freeze=freeze)


    optimizer, scheduler = initialize_model(bertClassifier, device =device, lenght_train_loader = lenght_train_loader, epochs=4)

    return bertClassifier, optimizer, scheduler

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

def elmoBilstm_creator(embed_dim = 1024,
            hidden_size = 100,
            n_classes = 3,
            n_lstm_layers = 1,
            dropout = 0.5,
            device = None,
            bidirectional = True,
            elmo = None):

    model = elmoBilstm(embed_dim = embed_dim,
            hidden_size = hidden_size,
            n_classes = n_classes,
            n_lstm_layers = n_lstm_layers,
            dropout = dropout,
            device = device,
            bidirectional = bidirectional,
            elmo = elmo)
    return model.to(device)




