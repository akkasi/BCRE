import pandas as pd
from overSampling import RandomOversampling
from preprocessing import *
from modelTrainers import *
from dataLoader import *
import re, random
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from multiView_BioBert import *
from transformers import BertForSequenceClassification, AdamW,get_linear_schedule_with_warmup
config = yaml.load(config_file, Loader=yaml.FullLoader)
class fineTuning():
    def __init__(self):
        self.baseModel = None
        self.set_seed()
        self.Oversampling = config['Oversampling']
        self.path_to_dataSet = config['path_to_dataset']
        self.train, self.validation, self.n_classes = self.readData()
        if self.Oversampling:
            self.train = RandomOversampling(self.train)


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.findBestPerformingModel()
        tokenizer = bertPreprocessor(self.baseModel)
        tokenizer.encode_data(self.train)
        tokenizer.encode_data(self.validation)



    def set_seed(self,seed_value=42):
        """Set seed for reproducibility.
        """
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def readData(self):
        data = pd.read_csv(self.path_to_dataSet, sep='\t', header=None, names=['text', 'label'])
        data.label.replace({'None': 0,
                            'E1 precedes E2': 1,
                            'E2 precedes E1': 2,
                            }, inplace=True)
        data.text = [re.sub(r'\s+', ' ', text).strip() for text in data.text.values]
        data.text = [x.lower() for x in data.text.values]
        data = data.reset_index(drop=True)

        X = data.text.values
        y = data.label.values
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)
        train, validation = pd.DataFrame(), pd.DataFrame()
        train['text'], validation['text'] = X_train, X_val
        train['label'], validation['label'] = y_train, y_val
        train.reset_index(drop=True, inplace=True)
        validation.reset_index(drop=True, inplace=True)
        return train, validation, len(set(data.label.values))

    def createModel(self, pre_model):
        model = BertForSequenceClassification.from_pretrained(
            pre_model,
            num_labels=self.n_classes,
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.

        )
        model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=2e-5,eps=1e-8)
        return model, optimizer

    def test_model(self, model):
        model.eval()
        predictions = list()
        groundTruth = list()

        validation_dataloader = bert_Dataloader(self.validation,batch_size=50)
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():

                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)


                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                predictions.extend(np.argmax(logits, axis=1).flatten())
                groundTruth.extend(label_ids)
        fscore = f1_score(groundTruth, predictions)
        return fscore, predictions, groundTruth

    def Evaluate(self,groundTruth, Predictions):
        print('Overall precision is: ',precision_score(groundTruth, np.array(Predictions, dtype=np.int64), average='micro'))
        print('Overall recall is: ', recall_score(groundTruth, np.array(Predictions, dtype=np.int64), average='micro'))
        print('Overall f-score is: ', f1_score(groundTruth, np.array(Predictions, dtype=np.int64), average='micro'))


    def findBestPerformingModel(self):

        bestPerformance = 0

        for pre_model in config['bioBertModels']:

            model = self.createModel(pre_model)
            fscore,_,_ = self.test_model(model)
            if fscore >= bestPerformance:
                bestPerformance = fscore
                self.baseModel = pre_model
        print('The best performing model is: '+ self.baseModel)

    def initialize_model(self, bertClassifier, epochs=4):

        bertClassifier.to(self.device)

        # Create the optimizer
        optimizer = AdamW(bertClassifier.parameters(),
                          lr=5e-5,  # Default learning rate
                          eps=1e-8  # Default epsilon value
                          )

        total_steps = len(bert_Dataloader(self.train,batch_size=config['batch_size'])) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=total_steps)
        return  optimizer, scheduler

    def fineTunner(self):
        loss_fn = nn.CrossEntropyLoss()
        bertClassifier = BertClassifier(self.baseModel, D = 768, n_C = self.n_classes,
                                        kernels = config['kernels'], n_kernels = config['n_kernels'][0],
                                        dropout=config['dropout'], freeze = True)
        optimizer, scheduler = self.initialize_model(bertClassifier, epochs=4)
        bert_trainer(bertClassifier, bert_Dataloader(self.train,batch_size=config['batch_size']),
                     loss_fn,optimizer, scheduler,
                     self.device, val_dataloader=None,
                     epochs=4, evaluation=False)
        predictions = bert_predict(bertClassifier, self.validation, self.device)
        groundtruth = self.validation.label.values
        self.Evaluate(groundtruth,predictions)
        torch.save(bertClassifier,config['path_to_model_to_be_saved'])