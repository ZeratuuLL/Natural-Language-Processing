import torch
import torch.nn as nn
import torch.functional as F

class FastText(nn.Module):
    
    def __init__(self, embedding_size, word_size, class_num, pad_token):
        
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(word_size, embedding_size, pad_token)
        self.fc1 = nn.Linear(embedding_size, class_num)
        self.output = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(p = dropout_rate)
        
    def forward(self, sentences):
        
        embedded = self.embedding(sentences)
        with torch.no_grad():
            # number of effective words (remove <PAD>)
            word_count = (embedded.pow(2).sum(dim=-1)>0).sum(dim=-1).view(-1, 1).float()
        embedded = embedded.sum(dim = 1) / word_count
        embedded = self.dropout(embedded)
        logits = self.output(self.fc1(embedded))
        
        return logits