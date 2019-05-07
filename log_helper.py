import pickle
from matplotlib import pyplot as plt
class Logger(object):
    def __init__(self, timestamp):
        self.ts = timestamp
    def pr(self, txt):
        print(txt)
        self.write(txt)
    
    def write(self, txt):
        with open(f'../logs/{self.ts}.txt','a') as model_metadata:
            model_metadata.write(txt+'\n')

    def plot_loss(self, history):
        plt.figure(2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss {self.ts}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(f'../logs/{self.ts}.png')
        #plt.show()
        pickle.dump(history.history['val_loss'],open(f'../logs/{self.ts}.val_loss','wb'))