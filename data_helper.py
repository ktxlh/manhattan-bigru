import itertools
import json
from re import sub
import string

from pandas import read_csv
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split as split_data
from keras.preprocessing.sequence import pad_sequences as keras_pad

class Data(object):
    def __init__(self, data_file, update_vocab, sequence_length=None, mode='train', train_ratio=0.8):
        self.data_file = data_file
        self.update_vocab = update_vocab
        self.sequence_length = sequence_length
        self.mode = mode
        
        self.score_col = 'score'
        self.sequence_cols = ['sentence1', 'sentence2']

        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        
        if mode == 'train':
            self.train_ratio = train_ratio
        else: # 'test'
            self.train_ratio = 0

        if update_vocab:
            self.vocab2id = {'<PAD>':0}
        else:
            with open('../model/vocab2id.json', 'r') as f:
                self.vocab2id = json.loads(f.read())            
        self.vocab = set([key for key in self.vocab2id.keys()])
        self.vocab_size = len(self.vocab)
        
        self.split_data(self.load_data())
        self.pad_sequences()

    def text_to_tokens(self, text):
        filter_set = string.punctuation + r'\\\,\+\.\-'     #TODO: How about 0123456789 ?
        return [sub('\'','',token) 
                for token in word_tokenize( sub(f'[{filter_set}]', " ", text.lower()) )] #token not in set(stopwords.words('english'))
        # Removing stopwords may change the meaning: if token not in set(stopwords.words('english'))

    def load_data(self):
        data_df = read_csv(self.data_file, sep='\t', dialect='excel-tab') 
        
        OOV_count = 0
        word_count = 0
        # Iterate dataset
        for index, row in data_df.iterrows():
            # Iterate ['sentence1', 'sentence2']
            for sequence in self.sequence_cols:
                sentence_id = []
                tokens = self.text_to_tokens(str(row[sequence]))
                for token in tokens:
                    word_count += 1
                    if token in self.vocab:
                        sentence_id.append(self.vocab2id[token])
                    elif self.update_vocab:
                        self.vocab.add(token)
                        self.vocab2id[token] = len(self.vocab)-1
                        sentence_id.append(self.vocab2id[token])
                    else:
                        OOV_count += 1
                    
                data_df.at[index, sequence] = sentence_id
                
                if len(sentence_id) < 2:
                    print("BAD INPUT #######################################################")
                    print(index, row, sequence, len(sentence_id), len(tokens))
                    print(tokens)
                    print(sentence_id, end='\n\n\n')
        
        if self.update_vocab:
            with open('../model/vocab2id.json','w') as f:
                f.write(json.dumps(self.vocab2id))
        
        self.vocab_size = len(self.vocab)
        print(f"load_data Out-of-Vocab ratio = {(OOV_count+1e-7)/(word_count+1e-7):.5f}")
        return data_df

    def pad_sequences(self):
        if self.sequence_length == 0:
            self.sequence_length = max(max(len(seq) for seq in self.x_train[0]),
                               max(len(seq) for seq in self.x_train[1]),
                               max(len(seq) for seq in self.x_val[0]),
                               max(len(seq) for seq in self.x_val[1]))

        # Zero padding
        for dataset, side in itertools.product([self.x_train, self.x_val], [0, 1]):
            dataset[side] = keras_pad(dataset[side], maxlen=self.sequence_length)
        # keras_pad: If maxlen < len(sentence_id), it will take the last maxlen elements.

    def split_data(self, data_df):
        data_size = len(data_df)

        X = data_df[self.sequence_cols]
        Y = data_df[self.score_col]

        self.x_train, self.x_val, self.y_train, self.y_val = split_data(X, Y, train_size=self.train_ratio)
        
        self.x_train = [self.x_train[column] for column in self.sequence_cols]
        self.x_val = [self.x_val[column] for column in self.sequence_cols]
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values

    def clean_file(self):
        # One-off. For the bad dataset :(
        with open('../STS-B/test2.tsv','w') as in_f:
            with open('../STS-B/test.tsv','r') as out_f:
                in_f.write(out_f.read().replace('\"',''))
        with open('../STS-B/train2.tsv','w') as in_f:
            with open('../STS-B/train.tsv','r') as out_f:
                in_f.write(out_f.read().replace('\"',''))