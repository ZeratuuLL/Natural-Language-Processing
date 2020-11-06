import pandas as pd

import config

train = pd.read_csv(config.train_path)
test = pd.read_csv(config.test_path)

data = pd.concat([train[['Brand', 'Model']], test[['Brand', 'Model']]])

brands = list(set(data['Brand'].values) | set(data['Model'].values)) # the first value is nan
brands = brands[1:]

#write to file
with open('./data/brands.txt', 'w+') as f:
    for brand in brands:
        f.write(brand + '\n')
    f.close()