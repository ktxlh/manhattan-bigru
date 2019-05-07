from time import strftime, localtime, time
from datetime import timedelta

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, Lambda, concatenate
from keras import optimizers
#from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from data_helper import Data
from embedding_helper import Get_Embedding
from log_helper import Logger

TIME_STAMP  = strftime("%m%d%H%M%S", localtime())  # to avoid duplicated file names deleting files
TRAINING_DATA_PATH = '../STS-B/train2.tsv'
TESTING_DATA_PATH = '../STS-B/test2.tsv'
EMBEDDING_PATH = '../model/GoogleNews-vectors-negative300.bin'
lg = Logger(TIME_STAMP)

def build_model(hidden_size, drop, sequence_length, vocab2id, update_vocab):

    out_size = hidden_size * 1 # 2 if bi_sequential is 'concat'enated; 1 if it's 'sum'med, 'ave'raged...

    def exp_neg_manhattan_dist(x):
        return 5*K.exp(-K.sum(K.abs(x[:,:out_size] - x[:,out_size:]), axis=1, keepdims=True))

    print('Building Model')

    input_1 = Input(shape=(sequence_length,), dtype='int32')
    input_2 = Input(shape=(sequence_length,), dtype='int32')
    # input_ -> [batch_size, sequence_length]
    
    embedding = Get_Embedding(EMBEDDING_PATH, vocab2id, update_vocab)
    embedding_dim = embedding.matrix.shape[1]
    emb_layer = Embedding(input_dim=len(vocab2id),
                            output_dim=embedding_dim,
                            input_length=sequence_length, 
                            trainable=False,    # TODO
                            weights=[embedding.matrix])
    embedding_1 = emb_layer(input_1)
    embedding_2 = emb_layer(input_2)
    # embedding_ -> [batch_size, sequence_length, embedding_dim]

    ##################################################
    # Single-layer version: Due to little data       #
    # May be modified for multi-layer with iteration #
    ##################################################
    #penalty = 100
    rnn = Bidirectional(
            GRU(                # TODO: LSTM or GRU?
            units=hidden_size,          
            #kernel_regularizer=regularizers.l2(penalty),
            #bias_regularizer=regularizers.l2(penalty),
            #recurrent_regularizer=regularizers.l2(penalty),
            #activity_regularizer=regularizers.l1(penalty),
            dropout=drop, 
            recurrent_dropout=drop,
            return_sequences=False,     # return_sequence=False => Returns the last output only (For the last layer only in this work)
            unroll=True)
        ,merge_mode='sum')            # merge_mode: Mode by which outputs of the forward and backward RNNs will be combined. 
                                        # TODO: One of {'sum', 'mul', 'concat', 'ave', None}. 
                                        # If None, the outputs will not be combined, they will be returned as a list.
    
    rnn_out_1 = rnn(embedding_1)
    rnn_out_2 = rnn(embedding_2)

    concat_out = concatenate([rnn_out_1, rnn_out_2], axis=-1)

    lambda_out = Lambda(exp_neg_manhattan_dist, output_shape=(1,))(concat_out)

    model = Model(inputs=[input_1, input_2], outputs=[lambda_out])

    # Optimizers: Adam outperforms SGD in "deep" neural networks
    optimizer = optimizers.Adam()   #lr=1e-3?
    #optimizer = optimizers.SGD(lr=2e-4, clipnorm=1.)
    #optimizer = optimizers.SGD(lr=2e-4, nesterov=True, clipnorm=100)
    #optimizer = optimizers.Adadelta(clipnorm=1.) #1.25

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print(model.summary())
    return model

def train_model(train_ratio=0.9,
                update_vocab=True,
                batch_size = 32,
                epochs = 50,
                sequence_length = 61, # 61
                hidden_size = 100,    # Rule of thumb~=100 (Concat -> 50?)
                drop = 0.5):          # 0.2, 0.4 or 0.5
                
    print('Loading data.')
    data = Data(data_file=TRAINING_DATA_PATH, 
                update_vocab=update_vocab,
                sequence_length=sequence_length,
                mode='train', train_ratio=train_ratio,)
    sequence_length = data.sequence_length
    x_train = data.x_train
    y_train = data.y_train
    x_val = data.x_val
    y_val = data.y_val
    vocabulary_size = data.vocab_size

    print('\n')
    lg.pr(f'''# training samples        : {len(x_train[0])}
        # validation samples      : {len(x_val[0])}
        Maximum sequence length   : {sequence_length}
        Vocabulary Size           : {vocabulary_size}
        ''')

    model = build_model(
        hidden_size, drop, sequence_length, data.vocab2id, update_vocab)
    
    # Early Stopping
    model_fname = f'../model/{TIME_STAMP}-{epochs}ep-{hidden_size}hs-{drop*10:.0f}dp'
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2),
             ModelCheckpoint(filepath=model_fname, monitor='val_loss', save_best_only=True)]

    training_start_time = time()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
    #verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    lg.pr(f"Training time finished.\n{epochs} epochs in {timedelta(seconds=time()-training_start_time)}")
    
    lg.plot_loss(history)
    lg.pr(f"""Model saved. Name: \'{model_fname}\'
        train_ratio    :{train_ratio}
        update_vocab   :{update_vocab}
        batch_size     :{batch_size}
        epochs         :{epochs}
        sequence_length:{sequence_length}
        hidden_size    :{hidden_size}
        drop           :{drop}""")
    with open(f'../logs/{TIME_STAMP}.txt','a') as model_metadata:
        model.summary(print_fn=lambda x: model_metadata.write(x + '\n'))
    
    return model

def test_model(model=None, model_fname=''):
    if model == None:
        model = load_model(model_fname)

    data = Data(data_file=TESTING_DATA_PATH,
                update_vocab=False,
                sequence_length=61,
                mode='test', train_ratio=0.)
    x_test = data.x_val
    y_test = data.y_val
    y_pred = np.array(model.predict(x_test))[:,0]

    pearson_r, pearson_p = pearsonr(y_test, y_pred)
    spearman_rho, spearman_p = spearmanr(y_test, y_pred)
    score = pearson_r + spearman_rho
    mse = mean_squared_error(y_test, y_pred)

    print("\n")
    lg.pr(f"""# testing samples                           : {len(y_test)}
        Pearson correlation coefficient             : {pearson_r}
        Spearman rank-order correlation coefficient : {spearman_rho}
        Total score                                 : {score}
        Mean squred error                           : {mse}
        """)
    
    #print("Some results (real, predicted): ", [t for t in zip(y_test[:20], y_pred[:20])])


if __name__ == "__main__":
    model = train_model( train_ratio = 1-1e-1, update_vocab=False, hidden_size=30, epochs=1)
    test_model(model=model)
    #test_model(model_fname='../model/bilstm-20190417142455')
    