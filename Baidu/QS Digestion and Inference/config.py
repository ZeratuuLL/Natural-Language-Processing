from zhon.hanzi import punctuation as punc1
from string import punctuation as punc2
import pandas as pd

#paths
train_path = './data/AutoMaster_TrainSet.csv'
test_path = './data/AutoMaster_TestSet.csv'
new_train_path = './data/NEW_train.csv'
new_test_path = './data/NEW_test.csv'

word2vec_model_path = './wv/word2vec.model'
word2ind_dic_path = './wv/word2ind.pkl'

seq2seq_model_path = None


#special symbols
PUNCTUATION = punc1 + punc2 + ' '
VERBAL = '<VERBAL>' # token for [语音]
EMPTY = '<EMPTY>' # token for nan
PIC = '<PIC>' #token for [图片]
START = '<START>'
END = '<END>'
UNKNOWN = '<UNK>'
PAD = '<PAD>'
COLUMNS = ['Question', 'Dialogue', 'Report']

#Constants
ROW_TRAIN = 82871 # number of rows in training set
ROW_TEST = 20000
f = open('./data/stopwords.txt', 'r')
stop_words = [stop_word.strip() for stop_word in f.readlines()]
f.close()

#hyperparameters
MIN_COUNT = 3
WINDOW = 4
HIDDEN_SIZE = 300
NUM_WORKER = 10
REMOVE = list(PUNCTUATION) + [VERBAL, EMPTY, PIC, ''] + stop_words
ITER = 3