## Project Description

To be added

## Dataset

Chinese dataset is Dureader. Check [this link](https://ai.baidu.com/broad/introduction?dataset=dureader) for detailed description
English dataset is SQuAD 2.0. Please read [this paper](https://arxiv.org/abs/1806.03822) for better understanding

## Data Preprocessing

Current the only implemented data preprocessing is word extracting and counting. The code can be found in **preprocessing.py**. Now only words that appear in questions are counted. 

To count Chinese words, jieba is used to split sentences into lists of words. 

For English words, there are three options. 
  * 1. Remove all punctuations and split by blank spaces. 
  * 2. Use NLTK's tokenizer to split words
  * 3. Use stanze's english tokenizer to split words.
