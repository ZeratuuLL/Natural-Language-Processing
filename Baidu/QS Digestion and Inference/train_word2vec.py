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
WORD_COUNT_PATH = config.word_count_path
REMOVE = config.REMOVE

with open(config.word_count_path, 'rb') as f:
    counter = pickle.load(f)
    f.close()
    
with open(config.target_word_count_path, 'rb') as f:
    target_counter = pickle.load(f)
    f.close()
    
REMOVE = [word for word in REMOVE if target_counter[word]<config.TARGET_MIN_COUNT] + [' ']# add back the words which appears some times in Report column

sentences = []

for index, file_path in enumerate(REPORTS + QUESTIONS + DIALOGUES):
    with open('./data/' + file_path, 'r') as f:
        for line in f.readlines():
            line = line[:-1] # to remove '\n' at the end of each line
            if index<3:
                # directly process a single sentence
                sentence = [[word for word in line.split(' ') if word not in REMOVE]] # make it a list of list
            else:
                # first split the line to get each sentence
                sentence = [[word for word in string.split(' ') if word not in REMOVE] for string in line.split('|')]
            if len(sentence)>0:
                sentences += sentence
        f.close()
        
print('sentences all filtered') # just to tell you this part is over

word_list = [word for word in counter.keys() if counter[word]>=config.MIN_COUNT] + [config.UNKNOWN, config.START, config.END, config.PAD]
word_list = dict(zip(word_list, [1 for _ in word_list]))
sentences = [utils.clean_sentence(sentence, word_list, max_len=int(1e4), remove=REMOVE, add=False, pad=False) for sentence in sentences]

print(len(sentences)) # just to tell you this part is over

model = Word2Vec(sentences,
                 size=config.HIDDEN_SIZE,
                 min_count=config.MIN_COUNT, # this min_count is also used to select words in utils.clean_sentence
                 workers=config.NUM_WORKER,
                 window=config.WINDOW, 
                 iter=config.ITER)

model.save(MODEL_PATH)

word2ind = {word: index for index, word in enumerate(model.wv.index2word)}
with open(WORD2IND_PATH, 'wb') as f:
    pickle.dump(word2ind, f)
    f.close()

with open(WORD_COUNT_PATH, 'wb') as f:
    pickle.dump(counter, f)
    f.close()