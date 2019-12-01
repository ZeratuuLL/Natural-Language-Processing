import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pad_and_clip_sequence

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
        
    def forward(self, input_sequences):
        embedded = self.embedding(input_sequences)
        if self.gate_type == 'lstm':
            output, (h, c) = self.rnn_cell(embedded)
        else:
            output, h = self.rnn_cell(embedded)
        return output, h
    
    
class pytorch_decoder(nn.Module):
    '''
    This decoder assumes that the input is a batch of same-length sentences, support of attention will be added later
    '''
    
    def __init__(self, vocab_size, embedding_weights=None, gate_type='gru', hidden_size=128, frozen=True):
        '''
        vocab_size : the number of words
        embedding_size : the length of words' embedding vectors
        embedding_weights : the pretrained embedding for words with shape (vocab_size, embedding_size), optional
        gate_type : gate type for RNN structure
        hidden_size : length of hidden state for rnn cells
        n_layers : number of rnn layers
        '''
        super(pytorch_decoder, self).__init__()
        self.output = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.gate_type = gate_type
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
        
    def forward(self, input, hidden, cell=None):
        embedded = self.embedding(input)
        if self.gate_type == 'lstm':
            output, (h, c) = self.rnn_cell(embedded, (hidden, cell))
            output = self.softmax(self.output(output)).squeeze() # remove the second dim with length 1, which is seq length
            return output, h, c
        else:
            output, h = self.rnn_cell(embedded, hidden)
            output = self.softmax(self.output(output)).squeeze() # remove the second dim with length 1, which is seq length
            return output, h
    
## TODO : add attention class and seq2seq class

## TODO : counterparts in tensorflow, in another file
    
if __name__ == '__main__':
    
    import random
    
    # hyperparameters
    input_length = 20
    output_length = 6
    sentence_number = 5
    hidden_size = 10
    vocab_size = 100
    gate_type = 'gru'
    n_layers = 3
    bidirectional = True
    
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
    test_dec = pytorch_decoder(vocab_size, embedding_weights, gate_type, hidden_size)
    
    # test encoder
    output, hidden = test_enc(test_inputs)
    print('All outputs from encoder\'s last layer has size {}. #(batch_size, sequence_length, hidden_size*num_direction)'.format(output.size()))
    print('Encoded context from encoder\'s each layer has size {}. #(num_layers, num_sentence, hidden_size)\n'.format(hidden.size()))
    
    # test decoder
    print('Decoding start\n')
    if bidirectional:
        hidden = output[:, -1, :hidden_size].unsqueeze(0) # encoded context
    else:
        hidden = output[:, -1, :].unsqueeze(0) # encoded context
    test_output_start = [random.sample(all_words, 1) for _ in range(sentence_number)]
    output = torch.tensor(test_output_start, dtype=torch.long)
    decoded = torch.zeros((sentence_number, output_length), dtype=torch.long)
    cell = torch.FloatTensor(1, sentence_number, hidden_size).normal_(0, 2)
    for i in range(output_length):
        if gate_type=='lstm':
            output, hidden, cell = test_dec(output, hidden, cell)
        else:
            output, hidden = test_dec(output, hidden)
        _, output = output.topk(1) # top1 output
        output = output.long()
        decoded[:, [i]] = output
    print('Decoded sentences are :\n{}'.format(decoded))