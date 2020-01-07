'''
This file:
    loads sentences from saved files
    remove special symbols
    find suitable max_len and cut/pad sequences
    train word vectors
    save weights and index<-->word dictionary
    
This file trains Word2Vec with each independent sentence with <START>, <END> and <UNK> but not <PAD>. It should be a little bit different from training from concatenated sentences
'''

from gensim.models import Word2Vec
import pickle

import utils
import config

QUESTIONS = ['TRAIN_Question.txt', 'TEST_Question.txt']
DIALOGUES = ['TRAIN_Dialogue.txt','TEST_Dialogue.txt']
REPORTS = ['TRAIN_Report.txt']
MODEL_PATH = config.word2vec_model_path
WORD2IND_PATH = config.word2ind_dic_path

sentences = []

for index, file_path in enumerate(QUESTIONS + REPORTS + DIALOGUES):
    with open('./data/' + file_path, 'r') as f:
        for line in f.readlines():
            line = line[:-1] # to remove '\n' at the end of each line
            if index<3:
                # directly process a single sentence
                sentence = [[word for word in line.split(' ') if word not in config.REMOVE]] # make it a list of list
            else:
                # first split the line to get each sentence
                sentence = [[word for word in string.split(' ') if word not in config.REMOVE] for string in line.split('|')]
            if len(sentence)>0:
                sentences += sentence
        f.close()

counter = utils.count_words(sentences)
sentences = [utils.clean_sentence(sentence, counter, max_len=int(1e4), pad=False) for sentence in sentences]

print(len(sentences)) # just to tell you this part is over

model = Word2Vec(sentences,
                 size=config.HIDDEN,
                 min_count=config.MIN_COUNT, # this min_count is also used to select words in utils.clean_sentence
                 workers=config.NUM_WORKER,
                 window=config.WINDOW,
                 iter=config.ITER)

model.save(MODEL_PATH)

word2ind = {word: index for index, word in enumerate(wv_model.wv.index2word)}
with open(WORD2IND_PATH, 'w+') as f:
    pickle.dump(word2ind, f)
    f.close()
