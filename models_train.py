import pandas as pd
import numpy as np
import sklearn as sk

import ann_classification
import K_nearest_neighbours
import Decision_Tree

def dummy(x, y):
    mode = np.argmax(np.bincount(y))
    def predict(data):
        return np.repeat(mode, data.shape[0])
    return (predict, mode)

models = {
    "knn": K_nearest_neighbours.KNN,
    "ann": ann_classification.train,
    "tree":Decision_Tree.Tree,
    "dummy": dummy
}

data_x = pd.read_csv("data.csv")
data_y = data_x["class"]
del data_x["class"]
data_x = np.array(data_x)
data_y = np.array(data_y)

validator = sk.model_selection.StratifiedKFold(n_splits = 5)

results = {}

for name in models.keys():
    results[name] = list()

for train, vali in validator.split(data_x, data_y):
    for name, model_fun in models.items():
        print(name)
        predictor, meta = model_fun(data_x[train], data_y[train])
        result = predictor(data_x[vali])
        print(sk.metrics.confusion_matrix(data_y[vali], result))
        results[name].append(sk.metrics.accuracy_score(data_y[vali], result))

for name, ress in results.items():
    print("model ", name, ". mean: ", np.mean(ress), ", sd: ", np.std(ress))
