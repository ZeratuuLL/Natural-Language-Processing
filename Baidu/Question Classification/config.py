import torch

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

roots = {'history' : './data/百度题库/高中_历史/origin/', 
         'geology' : './data/百度题库/高中_地理/origin/',
         'politics' : './data/百度题库/高中_政治/origin/',
         'biology' : './data/百度题库/高中_生物/origin/'}

def load_stop_words(path):
    file = open(path, 'r', encoding='utf-8')
    stopwords = file.readlines()
    stopwords = [word.strip() for word in stopwords]
    return stopwords
stopwords = load_stop_words('./stopwords/stopwords2.txt')

# common hyper-parameters

max_feature = 10000
embedding_size = 300
bs = 64
epoch = 30
dropout_rate = 0.1
lr_schedule = {0:0.01, 10 : 0.005, 20 : 0.001}

# hyper-parameters for FastText

NGramRange = 1

# hyper-paraemters for TextCNN

window_size_list = [2, 2, 3, 3, 4, 4, 5, 6, 7]
