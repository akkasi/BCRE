import torch
import numpy as np
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
import yaml
config_file = open("configure.yaml")
config = yaml.load(config_file, Loader=yaml.FullLoader)
class Preprocessing():
    def __init__(self, df, fname, seq_len = 200):
        self.path_to_pretrainedWE = fname
        self.data = df
        self.Max_seq_len = seq_len


    def tokenize(self):

        self.data.text = [word_tokenize(x) for x in (self.data).text.values]
        max_len = 0
        tokenized_texts = []
        word2idx = {}

        # Add <pad> and <unk> tokens to the vocabulary
        word2idx['<pad>'] = 0
        word2idx['<unk>'] = 1

        # Building our vocab from the corpus starting from index 2
        idx = 2
        for tokenized_sent in self.data.text.values:


            # Add `tokenized_sent` to `tokenized_texts`
            tokenized_texts.append(tokenized_sent)

            # Add new token to `word2idx`
            for token in tokenized_sent:
                if token not in word2idx:
                    word2idx[token] = idx
                    idx += 1

            # Update `max_len`
            max_len = max(max_len, len(tokenized_sent))

        return tokenized_texts, word2idx, max_len

    def encode(self, tokenized_texts, word2idx):
        tokenized_texts = [list(x) for x in tokenized_texts]

        input_ids = []
        for tokenized_sent in tokenized_texts:
            # Pad sentences to max_len
            tokenized_sent += ['<pad>'] * (self.Max_seq_len - len(tokenized_sent))

            # Encode tokens to input_ids
            input_id = [word2idx[token] for token in tokenized_sent][:self.Max_seq_len]
            if(len(input_id) != self.Max_seq_len):
                print(tokenized_sent,len(tokenized_sent))
            
            input_ids.append(np.array(input_id))

        return np.array(input_ids,dtype=np.int64)

    def load_pretrained_vectors(self, word2idx):

        print("Loading pretrained vectors...")
        fin = open(self.path_to_pretrainedWE, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())

        # Initilize random embeddings
        embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
        embeddings[word2idx['<pad>']] = np.zeros((d,))

        # Load pretrained vectors
        count = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in word2idx:
                count += 1
                embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

        # print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

        return torch.tensor(embeddings), torch.tensor(embeddings).shape[1]

class bertPreprocessor():
    def __init__(self,bertModel):
        self.data = None
        self.tokenizer = BertTokenizer(bertModel,do_lower_case=True)
        self.input_ids = list()
        self.attention_masks = list()

    def encode_data(self,df):
        self.data = df
        for sent in self.data.text:
            encoded_snt = self.tokenizer.encode_plus(
                text= sent,
                add_special_tokens= True,
                max_length= config['maxlen'],
                pad_to_max_length = True,
                return_attention_mask= True
            )
            self.input_ids.append(encoded_snt.get('input_ids'))
            self.attention_masks.append(encoded_snt.get('attention_masks'))
        self.data['input_ids'] = self.input_ids
        self.data['attention_masks'] = self.attention_masks
        self.data.reset_index(drop=True, inplace=True)













