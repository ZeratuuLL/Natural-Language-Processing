'''
This file saves the buildings blocks for a seq2seq model
Some test cases are also provided. If this file is run as the main file you can see the test result
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import pickle
from gensim.models import Word2Vec

import config

device = config.DEVICE

class pytorch_encoder(nn.Module):
    '''
    This encoder assumes that the input is a batch of same-length sentences
    '''
    
    def __init__(self, vocab_size, embedding_weights=None, gate_type='gru', hidden_size=128, n_layers=2, dropout_rate=0, bidirectional=False, frozen=True):
        '''
        vocab_size : the number of words
        embedding_size : the length of words' embedding vectors
        embedding_weights : the pretrained embedding for words with shape (vocab_size, embedding_size), optional
        gate_type : gate type for RNN structure
        hidden_size : length of hidden state for rnn cells
        n_layers : number of rnn layers
        dropout_rate : dropout probability for rnn cells except for the last layer. Check pytorch document
        bidirectional : whether to use bidirectional RNN structure
        frozen : whether to freeze (not train) embedding
        '''
        super(pytorch_encoder, self).__init__()
        self.gate_type = gate_type.lower()
        if gate_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=bidirectional, dropout = dropout_rate, batch_first=True)
        elif gate_type.lower() == 'gru':
            self.rnn_cell = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=bidirectional, dropout = dropout_rate, batch_first=True)
        else:
            self.rnn_cell = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=bidirectional, dropout = dropout_rate, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding_weights is not None:
            self.embedding.weight = nn.Parameter(embedding_weights)
        self.embedding.weight.requires_grad = not frozen
        
    def forward(self, input_sequences, embedding=None):
        '''
        Inputs:
        =======
        input_sequences : size (bs, max_len)
        embedding : an embedding layer, if None then self.embedding will be used. 
                    It can be used to pass an outside embedding layer. 
                    This is necessary because we cannot use pytorch optimizers if self.embedding.weight.requires_grad = False
        
        Outputs:
        ========
        output : size (bs, max_len, hidden_size)
        h : size (num_layers, bs, hidden_size)
        '''
        if embedding is None:
            embedding = self.embedding
        embedded = embedding(input_sequences)
        if self.gate_type == 'lstm':
            output, (h, c) = self.rnn_cell(embedded)
        else:
            output, h = self.rnn_cell(embedded)
        return output, h
    

class Bahdanau_Attention(nn.Module):

    def __init__(self, hidden_size, bidirectional=False):
        super(Bahdanau_Attention, self).__init__()
        if bidirectional: # for encoder_outputs
            self.input_size = 2*hidden_size
        else:
            self.input_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, 2*hidden_size) # for encoder_outputs
        self.fc2 = nn.Linear(hidden_size, 2*hidden_size) # for hidden_states
        self.fc3 = nn.Linear(2*hidden_size, 1)
        
    def forward(self, encoder_outputs, hidden_state):
        '''
        Inputs:
        =======
        encoder_outputs : size (batch_size, seq_len, hidden_size), the hidden state of each sentence at each step from the last layer of encoder.
        hidden_state : size (1, batch_size, hidden_size), the hidden state of a certain step from decoder
        
        Outputs:
        ========
        probs : size (batch_size, seq_len, 1), the weight for each hidden state so that they can be combined as a new context vector for the decoder
        '''
        hidden_state = hidden_state.permute(1, 0, 2) # shape to (bs, 1, hidden_size)
        vec1 = self.fc1(encoder_outputs) # expected (bs, max_len, 2*hidden_size)
        vec2 = self.fc2(hidden_state) # expected (bs, 1, 2*hidden)
        scores = torch.tanh(vec1 + vec2) # (bs, max_len, 2*hidden)
        probs = F.softmax(self.fc3(scores), dim=1)# (bs, max_len, 1)
        
        return probs
    

class pytorch_decoder(nn.Module):
    '''
    This decoder assumes that the input is a batch of same-length sentences, support of attention will be added later
    '''
    
    def __init__(self, vocab_size, embedding_weights=None, gate_type='gru', hidden_size=128, frozen=True, attn=None):
        '''
        vocab_size : the number of words
        embedding_size : the length of words' embedding vectors
        embedding_weights : the pretrained embedding for words with shape (vocab_size, embedding_size), optional
        gate_type : gate type for RNN structure
        hidden_size : length of hidden state for rnn cells
        frozen : whether to train the embedding weights
        attn : attention module
        '''
        super(pytorch_decoder, self).__init__()
        self.output = nn.Linear(hidden_size, vocab_size)
        if attn is not None:
            self.attn_combine = nn.Linear(attn.input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.gate_type = gate_type
        self.attn = attn
        self.hidden_size = hidden_size
        if gate_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        elif gate_type.lower() == 'gru':
            self.rnn_cell = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        else:
            self.rnn_cell = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding_weights is not None:
            self.embedding.weight = nn.Parameter(embedding_weights)
        self.embedding.weight.requires_grad = not frozen
        
    def forward(self, input, hidden, enc_output, embedding=None, cell=None):
        '''
        Inputs:
        =======
        input : size (bs, 1) as torch.long, the tokens
        hidden : size (1, bs, hidden_size) as the previous hidden cell (output from previous step)
        enc_output : size (bs, max_len, hidden_size) or (bs, max_len, 2*hidden_size), the output features from the last layer of encoder
        cell : size (1, bs, hidden_size) cell for LSTM if necessary
        embedding : an embedding layer, if None then self.embedding will be used. 
                    It can be used to pass an outside embedding layer. 
                    This is necessary because we cannot use pytorch optimizers if self.embedding.weight.requires_grad = False
        
        Outputs:
        ========
        output : size (bs, 1, vocab_size) as the probability for each word here
        hidden : size (1, bs, hidden_size) as the hidden cell of this step
        cell : size (1, bs, hidden_size) as the cell info of this steo if using LSTM
        '''
        if embedding is None:
            embedding = self.embedding
        embedded = embedding(input)
        if self.attn is not None:
            prob = self.attn(enc_output, hidden)
            context = torch.sum(prob*enc_output, dim=1, keepdim=True)
            embedded = torch.cat([embedded, context], dim=-1)
            embedded = self.attn_combine(embedded)
        if self.gate_type == 'lstm':
            output, (h, c) = self.rnn_cell(embedded, (hidden, cell))
            output = self.softmax(self.output(output)).squeeze() # remove the second dim with length 1, which is seq length
            return output, h, c
        else:
            output, h = self.rnn_cell(embedded, hidden)
            output = self.softmax(self.output(output)).squeeze() # remove the second dim with length 1, which is seq length
            return output, h
       
class Seq2Seq(nn.Module):
    
    def __init__(self, vocab_size, start_token, end_token, embedding_weights=None, gate_type='gru', hidden_size=128, n_layers=2, attention=True, dropout_rate=0, teaching_rate=0.5, bidirectional=False, frozen=True):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.gate_type = gate_type
        self.teaching_rate = teaching_rate
        self.bidirectional = bidirectional
        
        self.encoder = pytorch_encoder(vocab_size, None, gate_type, hidden_size, n_layers, dropout_rate, bidirectional, False).to(device)
        if attention:
            self.attention = Bahdanau_Attention(hidden_size, bidirectional).to(device)
        else:
            self.attention = None
        self.decoder = pytorch_decoder(vocab_size, None, gate_type, hidden_size, False, self.attention).to(device)
        if embedding_weights is not None:
            self.embedding = nn.Embedding(vocab_size, hidden_size).to(device)
            self.embedding.weight = nn.Parameter(embedding_weights).to(device)
            self.embedding.weight.requires_grad = not frozen
        else:
            self.embedding = None
            
        self.set_optimizers(0.001)
        
        self.start_token = start_token.to(device)
        self.end_token = end_token.to(device)
        self.teaching_rate = teaching_rate
        
    def set_optimizers(self, learning_rate):
        self.embedding_optimizer = None
        if self.embedding is not None:
            if self.embedding.weight.requires_grad:
                self.embedding_optimizer = optim.Adam(self.embedding.parameters(), learning_rate)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), learning_rate)
        
    def optimizer_zero_grad(self):
        if self.embedding_optimizer is not None:
            self.embedding_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
    def optimizer_step(self, clip_size=None):
        if clip_size is not None:
            if self.embedding is not None:
                torch.nn.utils.clip_grad_norm_(self.embedding.parameters(),clip_size)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(),clip_size)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(),clip_size)
        if self.embedding_optimizer is not None:
            self.embedding_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def set_embedding(self, self_embedding_weights=None, self_update=False, encoder_embedding_weights=None, decoder_embedding_weights=None):
        if self_embedding_weights is not None:
            if self_embedding_weights.size() == self.embedding.weight.size():
                self.embedding.weight = nn.Parameter(self_embedding_weights).to(device)
                self.embedding.weight.requires_grad = self_update
            else:
                print('Size does not equal. Self update failed')
        if encoder_embedding_weights is not None:
            if encoder_embedding_weights.size() == self.encoder.embedding.weight.size():
                self.encoder.embedding.weight = nn.Parameter(encoder_embedding_weights).to(device)
                self.encoder.embedding.weight.requires_grad = True
            else:
                print('Size does not equal. Encoder update failed')
        if encoder_embedding_weights is not None:
            if decoder_embedding_weights.size() == self.decoder.embedding.weight.size():
                self.decoder.embedding.weight = nn.Parameter(decoder_embedding_weights).to(device)
                self.decoder.embedding.weight.requires_grad = True
            else:
                print('Size does not equal. Decoder update failed')
                
    def train(self, dataset, epochs=10, learning_rate=None, batch_size=32, clip_size=None):
        
        if learning_rate is not None:
            self.set_optimizers(learning_rate)
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.NLLLoss(reduce=False)
        common_embedding = self.embedding
        
        for epoch in range(1, epochs+1):
            
            loss = 0
            loss_log = []
            
            for batch, (X, y) in enumerate(loader):
                
                self.optimizer_zero_grad()
                
                X = X.to(device)
                y = y.to(device)
                
                enc_output, enc_hidden = self.encoder(X, common_embedding)
                dec_hidden = enc_hidden[[-1], :, :]
                
                notEnd = torch.ones(y.size(0)).to(device)
                cell = torch.zeros(y.size(0), self.hidden_size)
                
                loss = 0
                
                for index in range(1, y.size(1)-1):
                    dec_input = y[:, [index]]
                    if self.gate_type == 'lstm':
                        dec_output, dec_hidden, cell = self.decoder(dec_input, dec_hidden, enc_output, common_embedding, cell)
                    else:
                        dec_output, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output, common_embedding)
                    losses = criterion(dec_output, y[:, index+1])
                    loss += (notEnd * losses).sum()
                    
                    notEnd = notEnd * ((y[:, index+1] != self.end_token).float().view(-1).to(device))
    
                    if not any(notEnd):
                        break
                    
                loss /= y.size(0)
                loss_log.append(loss.item())
                
                loss.backward()
                self.optimizer_step(clip_size)
                
                print('\rEpoch {} Batch {}, training average loss {}'.format(epoch, batch, loss.item()), end='')
            
            print('\tEpoch {} average loss {:.4f}'.format(epoch, np.mean(loss_log)))
            torch.save(self.state_dict(), config.seq2seq_model_path)
    
# A test case below    
if __name__ == '__main__':
    
    # hyperparameters
    input_length = 33
    output_length = 11
    sentence_number = 5
    hidden_size = 20
    vocab_size = 100
    gate_type = 'gru'
    n_layers = 3
    bidirectional = False
    
    # output status
    print('Testing with : ')
    print('\tinput sequence length : {}'.format(input_length))
    print('\toutput sequence length : {}'.format(output_length))
    print('\tnumber of test sentences : {}'.format(sentence_number))
    print('\tlenght of hidden units : {}'.format(hidden_size))
    print('\ttotal vocab size : {}'.format(vocab_size))
    print('\tencoder layers : {}'.format(n_layers))
    print('\tusing gate type : {}'.format(gate_type))
    print('\tbidirectional encoder activated : {}\n'.format(bidirectional))
    all_words = list(range(vocab_size))
    
    # create fake data
    test_inputs = [random.sample(all_words, input_length) for _ in range(sentence_number)]
    test_inputs = torch.tensor(test_inputs, dtype=torch.long)
    print('Test sentences size : {}\n'.format(test_inputs.size()))
    
    # embedding, encoder, decoder
    embedding_weights = torch.FloatTensor(vocab_size, hidden_size).uniform_(-1, 1)
    test_enc = pytorch_encoder(vocab_size, embedding_weights, gate_type, hidden_size, n_layers, 0, bidirectional)
    
    # test encoder
    enc_output, enc_hidden = test_enc(test_inputs)
    print('All outputs from encoder\'s last layer has size {}. #(num_sentence, sequence_length, hidden_size*num_direction)'.format(enc_output.size()))
    print('Encoded context from encoder\'s each layer has size {}. #(num_layers, num_sentence, hidden_size*num_direction)\n'.format(enc_hidden.size()))
    
    # test attention
    test_output_start = [random.sample(all_words, 1) for _ in range(sentence_number)]
    start = torch.tensor(test_output_start, dtype=torch.long)
    attn_hidden = enc_hidden[[-1], :, :]
    print(enc_hidden.size(), enc_output.size(), attn_hidden.size())
    
    test_attn = Bahdanau_Attention(hidden_size, bidirectional)
    prob = test_attn(enc_output, attn_hidden)
    print(prob.size(), enc_output.size())
    
    real_hidden = torch.sum(prob*enc_output, dim=1, keepdim=True)
    print('hidden vector after attention size : {}\n'.format(real_hidden.size()))
    
    # test decoder without attention
    test_dec = pytorch_decoder(vocab_size, embedding_weights, gate_type, hidden_size)
    print('Decoding start\n')
    dec_hidden = enc_hidden[[-1], :, :]
    test_output_start = [random.sample(all_words, 1) for _ in range(sentence_number)]
    dec_output = torch.tensor(test_output_start, dtype=torch.long)
    decoded = torch.zeros((sentence_number, output_length), dtype=torch.long)
    decoded[: , [0]] = torch.tensor(test_output_start, dtype=torch.long)
    cell = torch.FloatTensor(1, sentence_number, hidden_size).normal_(0, 2)
    for i in range(1, output_length):
        if gate_type=='lstm':
            dec_output, dec_hidden, cell = test_dec(dec_output, dec_hidden, enc_output, cell)
        else:
            dec_output, dec_hidden = test_dec(dec_output, dec_hidden, enc_output)
        _, dec_output = dec_output.topk(1) # top1 output
        dec_output = dec_output.long()
        decoded[:, [i]] = dec_output
    print('Decoded sentences are :\n{}'.format(decoded))
    
    # test decoder with attention
    test_attn = Bahdanau_Attention(hidden_size, bidirectional)
    test_dec = pytorch_decoder(vocab_size, embedding_weights, gate_type, hidden_size, attn=test_attn)
    print('Decoding start\n')
    dec_hidden = enc_hidden[[-1], :, :]
    test_output_start = [random.sample(all_words, 1) for _ in range(sentence_number)]
    dec_output = torch.tensor(test_output_start, dtype=torch.long)
    decoded = torch.zeros((sentence_number, output_length), dtype=torch.long)
    decoded[: , [0]] = torch.tensor(test_output_start, dtype=torch.long)
    cell = torch.FloatTensor(1, sentence_number, hidden_size).normal_(0, 2)
    for i in range(1, output_length):
        if gate_type=='lstm':
            dec_output, dec_hidden, cell = test_dec(dec_output, dec_hidden, enc_output, cell)
        else:
            dec_output, dec_hidden = test_dec(dec_output, dec_hidden, enc_output)
        _, dec_output = dec_output.topk(1) # top1 output
        dec_output = dec_output.long()
        decoded[:, [i]] = dec_output
    print('Decoded sentences are :\n{}'.format(decoded))
    
    # test Seq2Seq model
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
    
    dataset = TensorDataset(X[:500, :30], y[:500, ])
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print('Seq2Seq training started!')
    model.train(dataset, epochs=10, clip_size=5)