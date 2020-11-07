import numpy as np
import pandas as pd
import nltk
import jieba
import json
import tqdm
import re
import zhon.hanzi
import string

from collections import Counter, defaultdict

class Dureader_processor:
    
    def __init__(self, worddict_loc, stopwords_loc=None):
        '''
        Initialize class for preprocessing Dureader robust dataset
        
        Params:
        =======
        worddict_loc : string, the location to save word count dictionary
        stopwords_los : string, default None, the location for chinese stopwords file
        '''
        
        self.stopwords = defaultdict(lambda : 1) # default 1, stopwords 0
        self.worddict_loc = worddict_loc
        
        if stopwords_loc is not None:
            with open(stopwords_loc) as f:
                while True:
                    line = f.readline()
                    line = line.strip()
                    self.stopwords[line] = 0
                    if len(line) == 0:
                        break
                    
    def preprocessing(self, dataset_loc):
        '''
        Preprocessing the dataset. Currently this process includes:
            1. Extract context
            2. Remove punctuations with zhon.hanzi.punctuation and string.punctuation
            3. Split sentences into words by jieba.cut
            4. Remove stopwords
            5. Count words
            6. Save to desinated location
        '''
        
        with open(dataset_loc) as f:
            data = json.load(f)
        
        data = data['data'][0]['paragraphs']
        words = []
        for QA in tqdm.tqdm(data):
            context = QA['context']
            context = re.sub('[{}]'.format(zhon.hanzi.punctuation), ' ', context)
            context = re.sub('[{}]'.format(string.punctuation), ' ', context)
            for word in jieba.cut(context):
                if word != ' ' and self.stopwords[word]:
                    words.append(word)
                    
        count = Counter(words)
        
        with open(self.worddict_loc, 'w') as f:
            for word, count in count.most_common(len(count)):
                f.write('{} {}\n'.format(word, count))
            f.close()

#main running part
if __name__ == '__main__':
    dureader = Dureader_processor(worddict_loc='./chinese_word_count.txt', 
                                  stopwords_loc='./data/chinese_stopwords.txt')
    dureader.preprocessing('./data/dureader_robust-data/train.json')