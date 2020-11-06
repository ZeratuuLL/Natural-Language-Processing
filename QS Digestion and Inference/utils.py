'''
This file:
    provide basic functions for other files
'''

import pandas as pd
import jieba
from collections import Counter
import numpy as np
import math

import config

PUNCTUATION = config.PUNCTUATION
VERBAL = config.VERBAL # token for [语音]
EMPTY = config.EMPTY # token for nan
PIC = config.PIC #token for [图片]
START = config.START # token for start of sentence
END = config.END # token for end of sentence
UNKNOWN = config.UNKNOWN # token for unknown words
PAD = config.PAD # token for empty spaces
COLUMNS = config.COLUMNS
MIN_COUNT = config.MIN_COUNT
REMOVE = config.REMOVE

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
                        new_word = [word.strip() for word in new_word]
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
            new_word = [word.strip() for word in new_word]
            #new_word = [word for word in new_word if word not in punctuation] # remove all punctuations
            if len(new_word) == 0:
                new_word = [EMPTY]
            return new_word
    except (AttributeError, TypeError): # nan
        if split:
            return [[EMPTY, [EMPTY]]]
        else:
            return [EMPTY]
        
def process_columns(train, test, max_row=float('Inf')):
    '''
    preprocess one column, including:
    cut words, generate new columns with form [[words], [words], [words]...]
    also return the word set for this column
    this function should be customized for different dataset
    '''
    jieba.load_userdict('./data/brands.txt')
    
    preprocessed = dict(zip(['preprocessed_' + column for column in COLUMNS], [[] for _ in range(len(COLUMNS))]))
    for column in COLUMNS:
        if column == 'Dialogue':
            for index, sentence in enumerate(list(train.get(column, [])) + list(test.get(column, []))):
                if index >= max_row:
                    break
                new = process_sentence(sentence, True)
                preprocessed['preprocessed_' + column].append(new)
        else:
            for index, sentence in enumerate(list(train.get(column, [])) + list(test.get(column, []))):
                if index >= max_row:
                    break
                new = process_sentence(sentence, False)
                preprocessed['preprocessed_' + column].append(new)
    return preprocessed

def concat_dialogue(dialogue):
    '''
    a dialogue is a list of (speaker, list of words)
    This function extract all the list of words and concat them
    '''
    result = []
    for item in dialogue:
        result += list(item[1])
    return result

def clean_sentence(sentence, word_list, max_len=int(1e5), remove=REMOVE, add=False, pad=False):
    '''
    This function cleans a sentence:
        remove words that are not in word_list, for speed considertion, word_list is a dictionary with all values 1
        mask words that appear rarely with unknown words
        add START, END and PAD token
    '''
    sentence = [word for word in sentence if word not in remove]
    sentence = sentence[:max_len]
    sentence = [word if word_list.get(word, 0) else UNKNOWN for word in sentence]
    if add:
        sentence = [START] + sentence + [END]
    if pad:
        sentence += [PAD]*(max_len+2-len(sentence))
    return sentence

def get_maxlen(lens):
    '''
    Calculates the allowed max length for a group of sentences
    '''
    return math.ceil(np.mean(lens) + 2*np.std(lens))

def tokenize(sentence, word2ind):
    '''
    Transform sentences of words to lists of integers
    '''
    return [word2ind[word] for word in sentence]