import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics

import ann_classification

models = {
    "ann": ann_classification.train,
}

data_x = pd.read_csv("data.csv")
data_y = data_x["class"]
del data_x["class"]
data_x = np.array(data_x)

validator = StratifiedKFold(n_splits = 5)

results = {}

for name in models.keys():
    results[name] = list()

for train, vali in validator.split(data_x, data_y):
    for name, model_fun in models.items():
        predictor, meta = model_fun(data_x[train], data_y[train])
        result = predictor(data_x[vali])
        results[name].append(metrics.accuracy_score(data_y[vali], result))
