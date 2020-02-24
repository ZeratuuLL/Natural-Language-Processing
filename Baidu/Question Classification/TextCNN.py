import torch
import torch.nn as nn
import torch.functional as F

from config import device

class TextCNN(nn.Module):
    
    def __init__(self, embedding_weight, window_size_list, word_size, num_classes, pad_token, dropout_rate = 0.1, embedding_size = 300):
        '''
        embedding_weight : the fixed embedding
        window_size_list : a list choosing window sizes for CNN
        word_size : number of vocabulary
        num_classes : number of classes to classify
        pad_token : the token for <PAD>
        '''
        
        super(TextCNN, self).__init__()
        self.fixed_embedding = torch.tensor(embedding_weight).to(device)
        assert self.fixed_embedding.size(0) == word_size and self.fixed_embedding.size(1) == embedding_size
        self.fixed_embedding[pad_token] = 0

        self.embedding = nn.Embedding(word_size, embedding_size, pad_token)
        self.CNN_list = []
        for window_size in window_size_list:
            self.CNN_list.append(nn.Conv2d(2, 1, (window_size, embedding_size)).to(device))
        self.fc = nn.Linear(len(window_size_list), num_classes)
        self.output = nn.LogSoftmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout_rate)
        
    def forward(self, sentences):
        
        fixed = torch.index_select(self.fixed_embedding, 0, sentences.reshape(-1))
        fixed = fixed.reshape(sentences.size(0), sentences.size(1), -1)
        fixed = fixed.unsqueeze(1)
        
        embedded = self.embedding(sentences)
        embedded = embedded.unsqueeze(1) # add in_channel into shape
        embedded = torch.cat([fixed, embedded], 1)
        
        feature_list = []
        for cnn_layer in self.CNN_list:
            features = cnn_layer(embedded) # the last dimension should be 1
            features = features.squeeze() # remove the channel dimension and the last dimension
            features = torch.tanh(features) # activation layer
            features, _ = features.max(dim = -1) # MaxPooling
            feature_list.append(features)
            
        features = torch.stack(feature_list, dim = -1) # now shape is bs, max_window_size
        features = self.dropout(features) # dropout for normalization
        
        logits = self.fc(features)
        logits = self.output(logits)
        
        return logits