from utils import *
from config import *
from TextCNN import *

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

def train_TextCNN(subject):
    
    print('Reading Data')
    root = roots[subject]
    dataset = build_dataset(root)
    num_topics = len(dataset['label'].unique())
    common_texts = dataset['item'].tolist()
    
    print('Cleaning Data')
    common_texts, word2id, valid_words = filter_pad_words(common_texts, max_feature)
    id2word = dict(zip(word2id.values(), word2id.keys()))
    origin_texts = [[id2word[ind] for ind in sentence] for sentence in common_texts]
    
    print('Training Word2Vec')
    model = Word2Vec(origin_texts,
                     size = embedding_size,
                     min_count = 1, # this min_count is also used to select words in utils.clean_sentence
                     workers = 3,
                     window = 5, 
                     iter = 3)
    
    print('Feeding weights')
    fixed = np.zeros((len(word2id), embedding_size))
    for word, ind in word2id.items():
        fixed[ind] = np.array(model.wv[word])
    fixed = torch.from_numpy(fixed).float()
    
    Network = TextCNN(fixed, window_size_list, len(word2id), num_topics, len(word2id)-1, dropout_rate, embedding_size).to(device)
    optimizer = optim.Adam(Network.parameters(), lr_schedule[0])
    
    print('Creating training/testing set')
    label2id = dict(zip(dataset['label'].unique(), range(num_topics)))
    id2label = dict(zip(label2id.values(), label2id.keys()))
    X = np.array(common_texts)
    y = np.array([label2id[label] for label in dataset['label']]).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.2, 
                                                        random_state = 101)
    
    X_train = torch.tensor(X_train).long()
    y_train = torch.tensor(y_train).long()
    X_test = torch.tensor(X_test).long()
    y_test = torch.tensor(y_test).long()
    train = TensorDataset(X_train, y_train)
    test = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train, 64, True)
    test_loader = DataLoader(test, 64, False)
    
    print('Training\n')
    criterion = nn.NLLLoss()
    Network = Network.to(device)
    Network.train()
    for i in range(1, epoch + 1):
        
        log = []
        
        for X_sample, y_sample in iter(train_loader):
            
            X_sample = X_sample.to(device)
            y_sample = y_sample.view(-1).to(device)
            logits = Network(X_sample)
            loss = criterion(logits, y_sample)
            log.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Epoch {}. Average loss {:.4f}'.format(i, np.mean(log)))
        
        if i in lr_schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[i]
        
    print('\nTesting\n')
    predictions = []
    Network.eval()
    with torch.no_grad():
        
        for X_sample, _ in iter(test_loader):
            
            X_sample = X_sample.to(device)
            logits = Network(X_sample)
            _, index = logits.topk(1, 1)
            index = index.view(-1).cpu().numpy().tolist()
            predictions += index
    
    y_test = y_test.reshape(-1).tolist()
    y_test = [id2label[ind] for ind in y_test]
    predictions = [id2label[ind] for ind in predictions]
    
    print('\nTest result for {} :'.format(subject))
    print(classification_report(y_test, predictions))
    
    return TextCNN

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='FastText Subject')
    
    parser.add_argument('-s', '--subject', type=str, help='name of subject')
    args = parser.parse_args()
    
    _ = train_TextCNN(args.subject)