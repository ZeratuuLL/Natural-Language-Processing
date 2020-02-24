from config import *

import os
import re
import pandas as pd
import numpy as np
import jieba

from collections import Counter

def read_files(root):
    '''
    This function reads in all csv files lies directly under the root directory
    
    Returns the file directories as well as class names (file names)
    '''
    file_names = os.listdir(root)
    file_names = [name for name in file_names if name.endswith('csv')]
    classes = [name.split('.')[0] for name in file_names]
    file_names = [root + name for name in file_names]
    datasets = [pd.read_csv(name) for name in file_names]
    return datasets, classes

def clean_line(line):
    '''
    This function cleans the context
    '''
    line = re.sub(
            "[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+|题目", '',line)
    tokens = jieba.cut(line, cut_all=False)
    tokens = [token for token in tokens if token not in stopwords]
    return " ".join(tokens)

def build_dataset(root):
    
    datasets, classes = read_files(root)
    
    for dataset, label in zip(datasets, classes):
        dataset['item'] = dataset['item'].apply(lambda x:clean_line(x))
        dataset['label'] = label
    
    dataset = pd.concat(datasets, ignore_index = True)
    dataset = dataset[['item', 'label']]
        
    return dataset

def sentence_proc(sentence, max_len, word2id):
    
    if len(sentence) > max_len:
        sentence = sentence[:max_len]
    else:
        sentence += ['<PAD>'] * (max_len - len(sentence))
        
    sentence = [word2id.get(word, word2id['<OOV>']) for word in sentence]
    return sentence

def filter_pad_words(texts, max_feature):
    
    word_list = [word for sentence in texts for word in sentence]
    counter = Counter(word_list)
    counter = [(word, count) for word, count in counter.items()]
    counter.sort(key = lambda x : x[1], reverse = True)
    
    valid_words = [word for word, _ in counter[:max_feature]]
    word2id = dict(zip(valid_words, range(1, len(valid_words) + 1) ) )
    word2id['<OOV>'] = 0
    word2id['<PAD>'] = len(word2id)
    
    lens = [len(sentence) for sentence in texts]
    max_len = int(np.mean(lens) + 2 * np.std(lens))
    
    texts = [sentence_proc(sentence, max_len, word2id) for sentence in texts]
    
    return texts, word2id