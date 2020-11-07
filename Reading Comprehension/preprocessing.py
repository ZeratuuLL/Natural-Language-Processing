import json
import tqdm
import re
import string

import zhon.hanzi
import stanza
import nltk
import jieba

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
        
        self.stopwords = []
        self.worddict_loc = worddict_loc
        
        if stopwords_loc is not None:
            with open(stopwords_loc) as f:
                while True:
                    line = f.readline()
                    line = line.strip()
                    if len(line) == 0:
                        break
                    else:
                        self.stopwords.append(line)
        self.stopwords = set(self.stopwords)
                    
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
        
        data = data['data'][0]['paragraphs'] # index with 0 because data['data'] has only 1 element
        words = []
        for QA in tqdm.tqdm(data):
            context = QA['context']
            context = re.sub('[{}]'.format(zhon.hanzi.punctuation), ' ', context)
            context = re.sub('[{}]'.format(string.punctuation), ' ', context)
            for word in jieba.cut(context):
                if word != ' ' and word not in self.stopwords:
                    words.append(word)
                    
        count = Counter(words)
        
        with open(self.worddict_loc, 'w') as f:
            for word, count in count.most_common(len(count)):
                f.write('{} {}\n'.format(word, count))
            f.close()
            

class SQuAD_processor:
    
     def __init__(self, worddict_loc, stopwords_loc=None):
        '''
        Initialize class for preprocessing SQuAS dataset
        
        Params:
        =======
        worddict_loc : string, the location to save word count dictionary
        '''
        
        self.worddict_loc = worddict_loc
        
        self.stopwords = set(nltk.corpus.stopwords.words('English'))
            
     def preprocessing(self, dataset_loc, mode=None):
        '''
        Preprocessing the dataset. Currently this process includes:
            1. Extract context
            2. Remove punctuations or tokenize words, depending on `mode`
            4. Remove stopwords
            5. Count words
            6. Save to desinated location
            
        Params:
        =======
        dataset_loc : string, location to load dataset from
        mode : string, default None, can be 'nltk' or 'stanza':
            if None, all punctuations will be removed
            if 'nltk' or 'stanze', the tokenizer will be removed and then pure punctuations will be removed (time consuming, especially for 'stanze')
        '''
        
        if mode not in ['nltk', 'stanza']:
            mode = None
        
        with open(dataset_loc) as f:
            data = json.load(f)
        data = data['data']
        
        #get all index, so that we can locate all context with data[i]['paragraphs'][j]['context']
        N = len(data)
        index_set = []
        for i in range(N):
            for j, _ in enumerate(data[i]['paragraphs']):
                index_set.append((i, j))
        
        if mode == 'stanza':
            tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
        elif mode == 'nltk':
            tokenizer = nltk.tokenize.word_tokenize
        
        words = []
        for i, j in tqdm.tqdm(index_set):
            context = data[i]['paragraphs'][j]['context']
            context = context.lower()
            
            if mode == 'stanza': #use stanze to tokenize
                context = tokenizer(context)
                for word in context.iter_words():
                    word = word.text
                    for char in word:
                        if char.isalpha(): # filter words with only punctuations
                            if word not in self.stopwords:
                                words.append(word)
                            break
            
            elif mode == 'nltk': # use nltk to tokenize
                context = tokenizer(context)
                for word in context:
                    for char in word:
                        if char.isalpha(): # filter out words with only punctuations
                            if word not in self.stopwords:
                                words.append(word)
                            break
                            
            else: # remove punctuation and split with blanks
                context = re.sub('[{}]'.format(string.punctuation), ' ', context)
                context = context.split(' ')
                for word in context:
                    if len(word) > 0 and word not in self.stopwords:
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
    
    squad = SQuAD_processor('./nltk_English_word_count.txt')
    squad.preprocessing('./data/train-v2.0.json', 'nltk')
    squad = SQuAD_processor('./basic_English_word_count.txt')
    squad.preprocessing('./data/train-v2.0.json', None)
    squad = SQuAD_processor('./stanza_English_word_count.txt')
    squad.preprocessing('./data/train-v2.0.json', 'stanza')