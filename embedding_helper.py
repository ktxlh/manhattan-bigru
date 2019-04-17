# Using embedding from Google News

import numpy as np
from gensim.models import KeyedVectors
import pickle

class Get_Embedding(object):
    def __init__(self, file_path, vocab2id, update_vocab):
        self.size = 300 # Dimensionality of Google News' Word2Vec
        self.matrix = self.create_embed_matrix(file_path, vocab2id, update_vocab)

    def create_embed_matrix(self, file_path, vocab2id, update_vocab):
        if update_vocab:
            word2vec = KeyedVectors.load_word2vec_format(file_path, binary=True)
            matrix = np.zeros((len(vocab2id), self.size))

            oov_count = 0
            for word, i in vocab2id.items():
                if word in word2vec.vocab:
                    # words not found in embedding index will be all-zeros.
                    matrix[i] = word2vec.word_vec(word)
                else:
                    #print(word)
                    oov_count = oov_count+1
            print(f"Out-of-vocab ratio: {oov_count/len(vocab2id):.5f}")

            del word2vec
            pickle.dump(matrix, open('../model/embedding_matrix.pkl','wb'))

        else:   
            matrix = pickle.load(open('../model/embedding_matrix.pkl','rb'))
        
        return matrix
