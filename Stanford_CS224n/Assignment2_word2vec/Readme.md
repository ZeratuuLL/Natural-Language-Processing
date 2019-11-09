There are multiple files so I created a folder for it.

The files are:

  + **a2.pdf** is the document for questions and instructions 
  + **Assignment2.pdf** is my writing
  + **get_datasets.sh** can be used to download datasets necessary by calling 
```sh get_datasets.sh```
  + **env.yml** can be used to create a new environment for python but that's not necessary here. If you want you can call
```conda env create -f env.yml```
  + **run.py** is for the final training. 
  + **word2vec.py** consisits the calculation of gradients, with simple sanity check
  + **sgd.py** only requires one line of code to complete SGD optimization method, with simple sanity check
  + **saved_state_40000.pickle** is the final trained weights
  + **utils** contains some provided functions as well as my downloaded text data. So it's not necessary to download the datasets again.
