from gensim.models import Word2Vec
import pickle
import numpy as np
import torch

from models import *
import config
device = config.DEVICE

word2vec_model = Word2Vec.load(config.word2vec_model_path)
with open(config.word2ind_dic_path, 'rb') as f:
    Word2Ind = pickle.load(f)
    f.close()
    
Ind2Word = dict(zip(Word2Ind.values(), Word2Ind.keys()))
start_token =  torch.tensor(Word2Ind['<START>'], dtype=torch.long)
end_token = torch.tensor(Word2Ind['<END>'], dtype=torch.long)
vocab_size = len(Ind2Word)
gate_type = 'gru'
n_layers = config.N_LAYER

embedding_weights = np.array([list(word2vec_model[Ind2Word[i]]) for i in Ind2Word.keys()])
embedding_weights = torch.tensor(embedding_weights).to(device)

model = Seq2Seq(vocab_size, start_token, end_token, embedding_weights, gate_type, config.HIDDEN_SIZE, n_layers, True, 0, 1, False, True)

X = np.load(config.token_train_input_path)
y = np.load(config.token_train_target_path)
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

print('Seq2Seq training started!')
model.train(dataset, epochs=100, clip_size=5)