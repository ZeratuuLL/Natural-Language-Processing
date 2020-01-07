website (in Chinese) : https://aistudio.baidu.com/aistudio/competition/detail/3

## Description of Files:

  * **config.py** : a file of hyperparameters
  * **utils.py** : providing supporting functions
  * **get_brands.py** : extracts brands from data
  * **read_data.py** : read in data and reorganize data from strings to formatted lists of words. Saves processed data to both csv and txt files. The saved content is somehow different
  * **train_word2vec.py** : add <START>, <END> and <UNK> tokens to the sentences and train the initial Word2Vec weights with independent sentences. Save trained weights and the word<-->index dictionary

## Other things to mention
  
  * **About data** : source data can be found in the link above and the stopwords I used can be found [here](https://github.com/goto456/stopwords/blob/master/%E5%93%88%E5%B7%A5%E5%A4%A7%E5%81%9C%E7%94%A8%E8%AF%8D%E8%A1%A8.txt). The computed results are not uploaded due to size limit
