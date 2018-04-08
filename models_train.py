import pandas as pd
import numpy as np
import sklearn as sk

import ann_classification
import K_nearest_neighbours

models = {
    #"knn": K_nearest_neighbours.KNN,
    "ann": ann_classification.train,
}

data_x = pd.read_csv("data.csv")
data_y = data_x["class"]
del data_x["class"]
data_x = np.array(data_x)
data_y = np.array(data_y)

validator = sk.model_selection.StratifiedKFold(n_splits = 20)

results = {}

for name in models.keys():
    results[name] = list()

for train, vali in validator.split(data_x, data_y):
    for name, model_fun in models.items():
        predictor, meta = model_fun(data_x[train], data_y[train])
        result = predictor(data_x[vali])
        results[name].append(sk.metrics.accuracy_score(data_y[vali], result))
