import numpy as np
import json
from scipy.io import loadmat
from sklearn.model_selection import KFold

# load data
datas =['flags','foodtruck','CHD_49','emotions','yeast','birds',
            'Image','scene','VirusPseAAC','PlantPseAAC',
            'Enron','GnegativeGO']
index = datas[3]
dataset = loadmat("dataset/"+index+".mat")
print(dataset.keys())

X = dataset['data']
Y = dataset['target']

kf = KFold(n_splits=3, shuffle=True)
# store kf index
splits = []

for train_index, test_index in kf.split(X):
    splits.append({
        'train_indices': train_index.tolist(),
        'test_indices': test_index.tolist()
    })

# store index to json file
with open(f'dataset/{index}_3cv1.json', 'w') as f:
    json.dump(splits, f, indent=4)

print("Cross-validation splits have been saved.")
