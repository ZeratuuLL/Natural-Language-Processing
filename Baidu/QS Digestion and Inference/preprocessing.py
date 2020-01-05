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
PIC = '<PIC>' #token for [图片]

TRAIN = pd.read_csv('AutoMaster_TrainSet.csv')
TEST = pd.read_csv('AutoMaster_TestSet.csv')
ROW = TRAIN.shape[0]
COLUMNS = ['Question', 'Dialogue', 'Report']

#Extract all brands
data = pd.concat([TRAIN[['Brand', 'Model']], TEST[['Brand', 'Model']]])

brands = list(set(data['Brand'].values) | set(data['Model'].values)) # the first value is nan
brands = brands[1:]

with open('brands.txt', 'w+') as f:
    for brand in brands:
        f.write(brand + '\n')
    f.close()
    
jieba.load_userdict('brands.txt')
    
#Now we extract sentences from dataframes, save to new txts    
def process_sentence(sentence, split=False):
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
                    if value == '[图片]':
                        new_word = [PIC]
                    elif value == '[语音]':
                        new_word = [VERBAL]
                    else:
                        new_word = jieba.lcut(value)
                    #new_word = [word for word in new_word if word not in punctuation] # remove all punctuations
                    if len(new_word) == 0:
                        new_word = [EMPTY]
                    if len(keys)>=1 and key == keys[-1]: # the same person talk twice
                        values[-1] += new_word
                    else: # the different person speaks
                        keys.append(key)
                        values.append(new_word)
            return [list(item) for item in list(zip(keys, values))]
        else:
            new_word = jieba.lcut(sentence)
            #new_word = [word for word in new_word if word not in punctuation] # remove all punctuations
            if len(new_word) == 0:
                new_word = [EMPTY]
            return new_word
    except (AttributeError, TypeError): # nan
        if split:
            return [[EMPTY, EMPTY]]
        else:
            return [EMPTY]
        
def process_columns(max_row=float('Inf')):
    '''
    preprocess one column, including:
    cut words, generate new columns with form [[words], [words], [words]...]
    also return the word set for this column
    this function should be customized for different dataset
    '''
    
    preprocessed = dict(zip(['preprocessed_' + column for column in COLUMNS], [[] for _ in range(len(COLUMNS))]))
    for column in COLUMNS:
        if column == 'Dialogue':
            for index, sentence in enumerate(list(TRAIN.get(column, [])) + list(TEST.get(column, []))):
                if index >= max_row:
                    break
                new = process_sentence(sentence, True)
                preprocessed['preprocessed_' + column].append(new)
        else:
            for index, sentence in enumerate(list(TRAIN.get(column, [])) + list(TEST.get(column, []))):
                if index >= max_row:
                    break
                new = process_sentence(sentence, False)
                preprocessed['preprocessed_' + column].append(new)
    return preprocessed

preprocessed = process_columns()

for i, dialogue in enumerate(preprocessed['preprocessed_Dialogue']):
    new_dialogue = [dialogue[0]] # first (speaker, sentence) pair in dialogue
    for speaker, sentence in dialogue[1:]: # if there are others
        if speaker == new_dialogue[-1][-1][0]: # if the speaker is the speaker of last sentence
            new_dialogue[-1][-1][1] += sentence
        else:
            new_dialogue.append((speaker, sentence))  
    preprocessed['preprocessed_Dialogue'][i] = new_dialogue

for column in COLUMNS:
    f1 = open('TRAIN_{}.txt'.format(column), 'w+')
    f2 = open('TEST_{}.txt'.format(column), 'w+')
    print(column) # just a notification
    for index, sentence in enumerate(preprocessed['preprocessed_' + column]):
        if column == 'Dialogue':
            sentence = [' '.join(words[1]) for words in sentence]
            sentence = '|'.join(sentence)
        else:
            sentence = ' '.join(sentence)
        if index<ROW:
            f1.write(sentence + '\n')
        else:
            f2.write(sentence + '\n')
    f1.close()
    f2.close()
            
    