import pandas as pd
import jieba
from zhon.hanzi import punctuation as punc1
from string import punctuation as punc2
from collections import Counter
import itertools
from gensim.models import Word2Vec

punctuation = punc1 + punc2 + ' '

VERBAL = '<VERBAL>' # token for [语音]
EMPTY = '<EMPTY>' # token for nan

class preprocessor():
    '''
    This class saves all used preprocessing, data clearning as public methods
    __init__ method only reads in training and testing data and initialize with special tokens
    
    self.process_sentence():
        to process one single sentence/dialogue
    
    self.process_columns():
        transforms columns into lists of words and get the Word2Ind dictionary. 
        Processed lists will be stored in self.preprocessed. 
        Dictionary is self.Word2Ind. 
        Collection of all words is self.words
        self.appearance counts the number of how many records contains a certain word
        Naive word count is self.word_count 
    
    self.word2index():
        replaces words in self.preprocessed with index from self.Word2Ind, results saved as self.index
    
    self.cleaning(): 
        removes some words from the word list by rules or instructions
    
    self.train_word2vec(): 
        extracts all sentences and concatenate them into a list. 
        Then gensim.models.Word2Vec will be applied. 
        One can choose to save sentences and word vectors.
    '''
    
    def __init__(self, train_path, test_path, drop_na):
        self.data_train = pd.read_csv(train_path)
        self.data_test = pd.read_csv(test_path)
        if drop_na:
            self.data_train = self.data_train.dropna()
            self.data_train = self.data_train.dropna()
        self.words = [VERBAL, EMPTY] # this counts all words, initialized by special tokens
        
    def process_sentence(self, sentence, split=False):
        '''
        method to process one single sentence
        a sentence will be returned as [words in sentence]
        if it's dialogue, the return will be [[(speaker, words in sentence)], [(speaker, words in sentence)], ...]
        applied cleanings:
            1. if the dialogue has only one speaker, no '|'
            2. if the dialogue has emoji like -_-||| which will create false empty sentences
        '''
        try: 
            if split: # column Dialogue, need to split to several list
                if '|' not in sentence:
                    sentences = [sentence] # checked. This would solve the problem
                else:
                    sentences = sentence.split('|')
                keys = []
                values = []
                for sentence in sentences:
                    if len(sentence)>0: # because sometimes there will be ||| in sentences, which lead to empty sentence
                        key, value = sentence[:3], sentence[4:] # can't use .split('：')
                        new_word = jieba.lcut(value) if value != '[语音]' else [VERBAL]
                        #new_word = [word for word in new_word if word not in punctuation] # remove all punctuations
                        if len(new_word) == 0:
                            new_word = [EMPTY]
                        if len(keys)>=1 and key == keys[-1]: # the same person talk twice
                            values[-1] += new_word
                        else: # the different person speaks
                            keys.append(key)
                            values.append(new_word)
                        self.words += new_word
                return [list(item) for item in list(zip(keys, values))]
            else:
                new_word = jieba.lcut(sentence)
                #new_word = [word for word in new_word if word not in punctuation] # remove all punctuations
                if len(new_word) == 0:
                    new_word = [EMPTY]
                self.words += new_word
                return new_word
        except (AttributeError, TypeError): # nan
            if split:
                return [[EMPTY, EMPTY]]
            else:
                return [EMPTY]

    def process_columns(self, max_row=float('Inf')):
        '''
        preprocess one column, including:
        cut words, generate new columns with form [[words], [words], [words]...]
        also return the word set for this column
        this function should be customized for different dataset
        '''
        appearance = [set() for _ in range(self.data_train.shape[0] + self.data_test.shape[0])]
        columns = ['Question', 'Dialogue', 'Report']
        self.preprocessed = dict(zip(['preprocessed_' + column for column in columns], [[] for _ in range(len(columns))]))
        for column in columns:
            if column == 'Dialogue':
                for index, sentence in enumerate(list(self.data_train.get(column, [])) + list(self.data_test.get(column, []))):
                    if index >= max_row:
                        break
                    new = self.process_sentence(sentence, True)
                    self.preprocessed['preprocessed_' + column].append(new)
                    for _, sentence in new:
                        appearance[index] |= set(sentence)
            else:
                for index, sentence in enumerate(list(self.data_train.get(column, [])) + list(self.data_test.get(column, []))):
                    if index >= max_row:
                        break
                    new = self.process_sentence(sentence, False)
                    self.preprocessed['preprocessed_' + column].append(new)
                    appearance[index] |= set(new)
        appearance = [list(item) for item in appearance]
        self.word_count = Counter(self.words)
        self.words = list(self.word_count.keys())
        self.appearance = Counter(list(itertools.chain(*appearance)))
        self.cleaning(insert = [VERBAL, EMPTY])
    
    def word2index(self):
        '''
        create the dictionary from words to index
        '''
        columns = ['Question', 'Dialogue', 'Report']
        self.index = dict(zip(['index_' + column for column in columns], [[] for _ in range(len(columns))]))
        for column in columns:
            if column == 'Dialogue':
                self.index['index_' + column] = [[(speaker, [self.Word2Ind.get(word, -1) for word in text]) for speaker, text in sentence] for sentence in self.preprocessed['preprocessed_' + column]]
                self.index['index_' + column] = [[(speaker, [number for number in numbers is number!=-1]) for speaker, numbers in sentence] for sentence in self.index['index_' + column]]
                self.index['index_' + column] = [[(speaker, numbers if len(numbers)>0 else [self.Word2Ind[EMPTY]]) for speaker, numbers in sentence] for sentence in self.index['index_' + column]]
            else:
                self.index['index_' + column] = [[self.Word2Ind.get(word, -1) for word in text] for text in self.preprocessed['preprocessed_' + column]]
                self.index['index_' + column] = [[number for number in numbers if number!=-1] for numbers in self.index['index_' + column]]
                self.index['index_' + column] = [numbers if len(numbers)>0 else [self.word2Id[EMPTY]] for numbers in self.index['index_' + column]]
    
    def cleaning(self, remove=[], insert=[], max_count=100000, min_count=1, max_appearance=100000, min_appearance=2):
        '''
        remove words in remove
        add words in insert
        remove words whose counts are larger than max_count / smaller than min_count
        remove words whose appearance are larger than max_appearance / smaller than min_appearance
        
        then remove these words in self.preprocessed 
        more modifications to be added
        '''
        remove += [word for word in self.words if self.word_count[word]>max_count]
        remove += [word for word in self.words if self.word_count[word]<min_count]
        remove += [word for word in self.words if self.appearance[word]>max_appearance]
        remove += [word for word in self.words if self.appearance[word]<min_appearance]
        self.words = set(self.words)
        self.words -= set(remove)
        self.words |= set(insert)
        self.words = list(self.words)
        
        self.Word2Ind = dict(zip(self.words, range(len(self.words))))
        self.Ind2Word = dict(zip(range(len(self.words)), self.words))
        
        for item in self.preprocessed['preprocessed_Question']:
            item = [word for word in item if self.Word2Ind.get(word, None)]
        for item in self.preprocessed['preprocessed_Report']:
            item = [word for word in item if self.Word2Ind.get(word, None)]
        for item in self.preprocessed['preprocessed_Dialogue']:
            for dialogue in item:
                dialogue[1] = [word for word in dialogue[1] if self.Word2Ind.get(word, None)]
        
    def train_word2vec(self, sentences_path = None, vector_path = None, return_vectors=False):
        '''
        concatenate all sentences and train word vectors.
        If sentences_path is not None, all sentences will be saved into a txt
        If vector_path is not None, all word vectors will be saved into a csv 
        '''
        sentences = self.preprocessed['preprocessed_Question'] + \
                    self.preprocessed['preprocessed_Report'] + \
                    [speaks[1] for dialogue in self.preprocessed['preprocessed_Dialogue'] for speaks in dialogue]
        if sentences_path:
            with open(sentences_path, 'w+') as f:
                for sentence in sentences:
                    f.write(' '.join(sentence) + '\n')
                f.close()
        self.vectors = Word2Vec(sentences, min_count = 3, size = 200, workers = 6)
        if vector_path:
            data = pd.DataFrame({'Id' : list(range(len(self.vectors.wv.vocab))), 
                                 'Vector': [list(self.vectors.wv[word]) for word in self.vectors.wv.vocab.keys()]},
                                index = list(self.vectors.wv.vocab.keys()))
            data.to_csv(vector_path)
        if return_vectors:
            return self.vectors
        
def pad_and_clip_sequence(sequences, max_len, start_id, end_id, pad_id):
    sequences = [[start_id]+sequence[:max_len]+[end_id]+[pad_id]*max(0, max_len-2-len(sequence)) for sequence in sequences]
    return sequences
        
        
if __name__ == '__main__':
    import time 
    
    TRAIN = 'AutoMaster_TrainSet.csv'
    TEST = 'AutoMaster_TestSet.csv'
    t1 = time.time()
    processor = preprocessor(TRAIN, TEST, True)
    
    processor.process_columns()
    
    processor.train_word2vec('sentences.txt', 'word_vectors.csv')
    print('Total preprocessing time : {}'.format(time.time() - t1))
    
    total_word_number = len(processor.word_count.keys())
    
    # BTW here are some statistics
    print('number of words appeared in only one record : {}/{}'.format(len([1 for value in processor.appearance.values() if value==1]), total_word_number))
    print('number of words whose counst are one : {}/{}'.format(len([1 for value in processor.word_count.values() if value==1]), total_word_number))
    print()
    print('number of words appeared in only two records : {}/{}'.format(len([1 for value in processor.appearance.values() if value==2]), total_word_number))
    print('number of words whose counst are two : {}/{}'.format(len([1 for value in processor.word_count.values() if value==2]), total_word_number))
    print()
    print('number of words appeared in less than five records : {}/{}'.format(len([1 for value in processor.appearance.values() if value<=5]), total_word_number))
    print('number of words whose counst are less than five : {}/{}'.format(len([1 for value in processor.word_count.values() if value<=5]), total_word_number))
    print()
    print('number of words appeared in less than ten records : {}/{}'.format(len([1 for value in processor.appearance.values() if value<=10]), total_word_number))
    print('number of words whose counst are less than ten : {}/{}'.format(len([1 for value in processor.word_count.values() if value<=10]), total_word_number))
    print()
    print('number of words appeared in more than ten thousand records : {}/{}'.format(len([1 for value in processor.appearance.values() if value>=10000]), total_word_number))
    print('number of words whose counst are more than ten thousand: {}/{}'.format(len([1 for value in processor.word_count.values() if value>=10000]), total_word_number))