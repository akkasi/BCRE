import pandas as pd
import yaml
import re
from sklearn.model_selection import StratifiedKFold
import numpy as np
from overSampling import RandomOversampling
from sklearn import metrics
from dataLoader import *
from Preprocessing import *
import torch.optim as optim
from modelCreators import *
from modelTrainers import *
import random
import tensorflow as tf
config_file = open("configure.yaml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

class crossValidation():
    def __init__(self, path_to_dataSet, model_type, path_to_outputs, Oversmpling=1, Folds = 10 ):
        self.path_to_outputs = path_to_outputs
        self.path_to_dataSet = path_to_dataSet
        self.data, self.classNo = self.readData()
        self.model_type = model_type
        self.folds = Folds
        self.Oversmpling = Oversmpling
        self.model = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.preprocessor = Preprocessing(self.data, config['path_to_pretrainedWE'], config['maxlen'])
        self.data.text, word2idx, _ = self.preprocessor.tokenize()
        self.embedding, self.embed_dim = self.preprocessor.load_pretrained_vectors(word2idx)
        self.data.text = [np.array(y) for y in self.preprocessor.encode(self.data.text.values, word2idx)]
        self.vocab_size = len(word2idx)
        self.set_seed()


    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        tf.random.set_seed(seed_value)

    def readData(self):
        data = pd.read_csv(self.path_to_dataSet, sep='\t', header=None, names=['text', 'label'])
        data.label.replace({'None': 0,
                            'E1 precedes E2': 1,
                            'E2 precedes E1': 2,
                            }, inplace=True)
        data.text = [re.sub(r'\s+', ' ', text).strip() for text in data.text.values]
        data.text = [x.lower() for x in data.text.values]
        return data, len(set(data.label.values))


    def createModel(self):

        if self.model_type == 'mvcc_g':
            self.model = mvcc_glove_creator(self.embedding,
                        config['freeze_embedding'],
                        self.vocab_size,
                        self.embed_dim,
                        config['kernels'],
                        config['n_kernels'],
                        self.classNo,
                        config['dropout'])

            self.model.to(self.device)
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=float(config['lr']))
            self.loss_fn = nn.CrossEntropyLoss()
            
        elif self.model_type == 'mvcc_bio':
            pass
        elif self.model_type == 'att_Bilstm_g':
            self.model = attn_BiLSTM(pretrained_embedding=self.embedding,
                        freeze_embedding=config['freeze_embedding'],
                        vocab_size=self.vocab_size,
                        embed_dim=self.embed_dim,
                        hidden_size= config['hidden_size'],
                        n_classes=self.classNo,
                        n_lstm_layers=config['n_lstm_layers'],
                        dropout=config['dropout'],
                        bidirectional = config['bidirectional'],
                        device=self.device)

            self.model.to(self.device)
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=float(config['lr']))
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.model_type == 'att_Bilstm_elmo':

            pass

    def train_model(self):
        if self.model_type in  ['mvcc_g', 'mvcc_bio', 'att_Bilstm_g' ]:
            torch_trainer(self.model, self.optimizer, self.train_loader, self.loss_fn, self.device, config['epochs'], val_dataloader=None )


        elif self.model_type == 'att_Bilstm_elmo':
            pass

        pass
    def test_model(self):
        if self.model_type in  ['mvcc_g', 'mvcc_bio', 'att_Bilstm_g' ]:
            (_, _, Predictions) = evaluate(self.model,  self.test_loader, self.loss_fn, self.device)


        elif self.model_type == 'att_Bilstm_elmo':
            pass

        return [Predictions[i].item() for i in Predictions]

    def writePredictions(self):
        if len(self.groundTruth) != len(self.Predictions):
            print('Number of predictions is different tham the number of ground truths!!!')
        else:
            Dict = {'Gold': self.groundTruth,'Predictions': self.Predictions}
            df = pd.DataFrame(Dict)
            df.to_csv(self.path_to_outputs, sep='\t')



    def Evaluate(self):
        print('Overall precision is: ',metrics.precision_score(self.groundTruth, np.array(self.Predictions, dtype=np.int64), average='micro'))
        print('Overall recall is: ', metrics.recall_score(self.groundTruth, np.array(self.Predictions, dtype=np.int64), average='micro'))
        print('Overall f-score is: ', metrics.f1_score(self.groundTruth, np.array(self.Predictions, dtype=np.int64), average='micro'))
        print(metrics.confusion_matrix(self.groundTruth, np.array(self.Predictions, dtype=np.int64)))

    def CV(self):
        self.Predictions = []
        self.groundTruth = []
        skf = StratifiedKFold(n_splits=self.folds)
        t = self.data.label
        i = 1
        for train_index, test_index in skf.split(np.zeros(len(t)), t):
            print('Running K = '+str(i)+' in K-fold Cross Validation....')

            if self.Oversmpling:
                train = RandomOversampling(self.data.loc[train_index])
            else:
                train = self.data.loc[train_index]
            test = self.data.loc[test_index]

            self.train_loader = data_loader(train, batch_size=config['batch_size'])
            self.test_loader = data_loader(test, batch_size=config['batch_size'])

            self.groundTruth.extend(test.label.values)
            self.createModel()

        #
            self.train_model()
        #     self.Predictions.extend(self.test_model())
            # self.model = None
            i += 1
        # #

        # print(type(self.groundTruth[1]), len(self.groundTruth),self.groundTruth)
        # print(type(self.Predictions[1]), len(self.Predictions),self.Predictions)
        # # self.writePredictions()
        # self.Evaluate()

crossV = crossValidation(path_to_dataSet = config['path_to_input'], model_type = config['model_type'],
                path_to_outputs = config['path_to_outputs'], Oversmpling = config['Oversmpling'], Folds = config['Folds'])

crossV.CV()







