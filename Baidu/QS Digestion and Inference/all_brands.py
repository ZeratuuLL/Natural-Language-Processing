import pandas as pd

train = pd.read_csv('AutoMaster_TrainSet.csv')
test = pd.read_csv('AutoMaster_TestSet.csv')

data = pd.concat([train[['Brand', 'Model']], test[['Brand', 'Model']]])

brands = list(set(data['Brand'].values) | set(data['Model'].values)) # the first value is nan
brands = brands[1:]

with open('brands.txt', 'w+') as f:
    for brand in brands:
        f.write(brand + '\n')
    f.close()