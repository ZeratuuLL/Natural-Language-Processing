'''
This file:
    reads in proprocessed data and get X and y
    apply cutting and padding
    update Word2Vec model weights
    save new data and new weights
'''

import config
import utils

import pandas as pd
import numpy as np
import pickle
import ast
from gensim.models import Word2Vec

REMOVE = config.REMOVE

# load results from previous calculation
model = Word2Vec.load(config.word2vec_model_path)
word_list = model.wv.index2word + [config.UNKNOWN, config.START, config.END, config.PAD]
word_list = dict(zip(word_list, [1 for _ in word_list]))

with open(config.word2ind_dic_path, 'rb') as f:
    Word2Ind = pickle.load(f)
    f.close()

with open(config.word_count_path, 'rb') as f:
    counter = pickle.load(f)
    f.close()

with open(config.target_word_count_path, 'rb') as f:
    target_counter = pickle.load(f)
    f.close()

REMOVE = [word for word in REMOVE if target_counter[word]<config.TARGET_MIN_COUNT] + [' ']
     
# read training set
train = pd.read_csv(config.new_train_path)
train['Question'] = train['Question'].apply(ast.literal_eval)
train['Dialogue'] = train['Dialogue'].apply(ast.literal_eval)
train['Report'] = train['Report'].apply(ast.literal_eval)
train['context'] = train.apply(lambda x : x[0] + x[1], axis=1)
train['context'] = train['context'].apply(lambda x : utils.clean_sentence(x, word_list, remove=config.REMOVE, add=False, pad=False))
train['Report'] = train['Report'].apply(lambda x : utils.clean_sentence(x, word_list, remove=REMOVE, add=False, pad=False))
print('Read training data')

# read test set
test = pd.read_csv(config.new_test_path)
test['Question'] = test['Question'].apply(ast.literal_eval)
test['Dialogue'] = test['Dialogue'].apply(ast.literal_eval)
test['context'] = test.apply(lambda x : x[0] + x[1], axis=1)
test['context'] = test['context'].apply(lambda x : utils.clean_sentence(x, word_list, remove=config.REMOVE, add=False, pad=False))
print('Read test data')

# get max len of context and clean
input_maxlen = utils.get_maxlen(pd.concat([train['context'], test['context']]).apply(len))
train['context'] = train['context'].apply(lambda x : utils.clean_sentence(x, word_list, input_maxlen, remove=config.REMOVE, add=True, pad=True))
test['context'] = test['context'].apply(lambda x : utils.clean_sentence(x, word_list, input_maxlen, remove=config.REMOVE, add=True, pad=True))

# get max len of target and clean and save
train_target_maxlen = utils.get_maxlen(train['Report'].apply(len))
train['Report'] = train['Report'].apply(lambda x : utils.clean_sentence(x, word_list, train_target_maxlen, remove=REMOVE, add=True, pad=True))
print('Data padded')

train = train[['context', 'Report']]
train.to_csv(config.padded_train_path, encoding='utf_8_sig')
test = test[['context']]
test.to_csv(config.padded_test_path, encoding='utf_8_sig')
print('Padded data saved')

# update Word2Vec model and save
sentences = list(train['context']) + list(train['Report']) + list(test['context'])
model.build_vocab(sentences, update=True)
model.train(sentences, epochs=config.ITER, total_examples=model.corpus_count)
model.save(config.word2vec_model_path)
Word2Ind = {word: index for index, word in enumerate(model.wv.index2word)}
with open(config.word2ind_dic_path, 'wb') as f:
    pickle.dump(Word2Ind, f)
    f.close()

print('Word2Vec updated and saved')

# tokenize and save
train['context'] = train['context'].apply(lambda x : utils.tokenize(x, Word2Ind))
test['context'] = test['context'].apply(lambda x : utils.tokenize(x, Word2Ind))
train['Report'] = train['Report'].apply(lambda x : utils.tokenize(x, Word2Ind))
np.save(config.token_train_input_path, np.array(list(train['context'])))
np.save(config.token_train_target_path, np.array(list(train['Report'])))
np.save(config.token_test_input_path, np.array(list(test['context'])))
print('Tokenized data saved')
