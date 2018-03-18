import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import gensim
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import os

# Data source path
data_source_path = "../../DATA/"
EPOCHS = 10000
MIN_COUNT = 5
SIZE = 100
WINDOW = 4
ALPHA = 0.075
NGRAMS = 1
class EpochSaver(CallbackAny2Vec):
    "Callback to save model after every epoch"
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0
        self.epoch_duration = 0.0
        self.epoch_start_time = time.time()


    def on_epoch_end(self, model):
        output_path = '{}_epoch_{} '.format(self.path_prefix, self.epoch)
        if(self.epoch % 3000 == 0):
            model.wv.save_word2vec_format(fname = data_source_path+'input_train/'+strategy+'/'+output_path+'fr_posos.bin',
                                          fvocab=data_source_path+'input_train/'+strategy+'/'+output_path+'fr_vocab_posos.txt',
                                          binary=True)

        self.epoch_duration = (time.time() - self.epoch_start_time)/(1+self.epoch)
        REMAINING_HOURS = int((EPOCHS-self.epoch)* self.epoch_duration / 3600)
        REMAINING_MINUTES = int(((EPOCHS-self.epoch)* self.epoch_duration - REMAINING_HOURS*3600)/60)
        print("Epoch {}, model saved to {}, {} hour(s) {} minutes ".format(self.epoch,
                                                                           output_path,
                                                                           REMAINING_HOURS,
                                                                           REMAINING_MINUTES),end='\r')
        self.epoch += 1


def train_embedding(strategy):
    X_train = pd.read_csv(data_source_path+'input_train/'+strategy+'/clean_input_to_train.csv', sep=";", index_col=0)
    X_test = pd.read_csv(data_source_path+'input_test/'+strategy+'/clean_input_to_test.csv', sep=";", index_col=0)

    sentences = [sentence.split() for sentence in X_train['question']]
    for sentence in X_test['question']:
        sentences.append(sentence.split())

    debut = time.time()

    epoch_saver = EpochSaver('temporary_model')
    scratch_model  = FastText(sentences,
                              size = SIZE,
                              alpha=ALPHA,
                              window=WINDOW,
                              word_ngrams=NGRAMS,
                              min_count=MIN_COUNT,
                              workers=os.cpu_count(),
                              iter=EPOCHS,
                              callbacks=[epoch_saver])
    fin = time.time()
    print(fin-debut)

    scratch_model.wv.save_word2vec_format(fname = data_source_path+'input_train/'+strategy+'/fr_posos.bin',
                                          fvocab=data_source_path+'input_train/'+strategy+'/fr_vocab_posos.txt',
                                          binary=True)

if __name__ == '__main__':
    # Import data
    strategies = ['stemming', 'stemming_no_accent'] #['soft', 'no_accent', 'stemming', 'stemming_no_accent']
    for strategy in strategies :
        train_embedding(strategy)
