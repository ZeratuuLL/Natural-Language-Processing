'''
This file:
    read in data
    find brands info and create customized dictionary
    process data into [list of words], without special start and end tokens and save
    count all words
    save the processed data into both csv and txt files.
'''

import pandas as pd
import jieba
import pickle
from gensim.models import Word2Vec
from collections import Counter

import utils
import config

COLUMNS = ['Question', 'Dialogue', 'Report']
TRAIN = pd.read_csv(config.train_path)
TRAIN.dropna(subset=COLUMNS, inplace=True)
TEST = pd.read_csv(config.test_path)
ROW = TRAIN.shape[0]

WORD_COUNT_PATH = config.word_count_path
TARGET_WORD_COUNT_PATH = config.target_word_count_path

jieba.load_userdict('./data/brands.txt')

#read and preprocess the data
preprocessed = utils.process_columns(TRAIN, TEST)

#get new csv files
preprocessed_TRAIN = pd.DataFrame(dict(zip(COLUMNS, [preprocessed['preprocessed_{}'.format(column)][:ROW] for column in COLUMNS])))
preprocessed_TRAIN['Dialogue'] = preprocessed_TRAIN['Dialogue'].apply(utils.concat_dialogue)
preprocessed_TRAIN.to_csv(config.new_train_path, encoding='utf_8_sig', index=False)
preprocessed_TEST = pd.DataFrame(dict(zip(COLUMNS[:2], [preprocessed['preprocessed_{}'.format(column)][ROW:] for column in COLUMNS[:2]])))
preprocessed_TEST['Dialogue'] = preprocessed_TEST['Dialogue'].apply(utils.concat_dialogue)
preprocessed_TEST.to_csv(config.new_test_path, encoding='utf_8_sig', index=False)
del preprocessed_TRAIN, preprocessed_TEST

#combine consecutive dialogues from a same person
for i, dialogue in enumerate(preprocessed['preprocessed_Dialogue']):
    new_dialogue = [dialogue[0]] # first (speaker, sentence) pair in dialogue
    for speaker, sentence in dialogue[1:]: # if there are others
        if speaker == new_dialogue[-1][-1][0]: # if the speaker is the speaker of last sentence
            new_dialogue[-1][-1][1] += sentence
        else:
            new_dialogue.append((speaker, sentence))  
    preprocessed['preprocessed_Dialogue'][i] = new_dialogue

#save new txt files and count words
counter = Counter()
target_counter = Counter()
for column in COLUMNS:
    f1 = open('./data/TRAIN_{}.txt'.format(column), 'w+')
    f2 = open('./data/TEST_{}.txt'.format(column), 'w+')
    print(column) # just a notification
    for index, sentence in enumerate(preprocessed['preprocessed_' + column]):
        if column == 'Dialogue':
            for speaker, words in sentence:
                counter.update(Counter(words))
            sentence = [' '.join(words[1]) for words in sentence]
            sentence = '|'.join(sentence)
        else:
            if column == 'Report':
                target_counter.update(Counter(sentence))
            counter.update(Counter(sentence))
            sentence = ' '.join(sentence)
        if index<ROW:
            f1.write(sentence + '\n')
        else:
            f2.write(sentence + '\n')
    f1.close()
    f2.close()
            
with open(WORD_COUNT_PATH, 'wb') as f:
    pickle.dump(counter, f)
    f.close()
    
with open(TARGET_WORD_COUNT_PATH, 'wb') as f:
    pickle.dump(target_counter, f)
    f.close()
    