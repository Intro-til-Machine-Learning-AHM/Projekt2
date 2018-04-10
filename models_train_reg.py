import pandas as pd
import numpy as np
import sklearn as sk

import linearreg_1_2
import ann_regression

models = {
    "lreg": linearreg_1_2.lreg,
    "ann": ann_regression.train
}

#data_x = pd.read_csv("data.csv")
#data_y = data_x["class"]
#del data_x["class"]
#data_x = np.array(data_x)
#data_y = np.array(data_y)

data = pd.read_csv('data.csv')
data_x = data.drop("class",axis=1).drop("insulin",axis=1)
data_y = data["insulin"]
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
        #print(sk.metrics.confusion_matrix(data_y[vali], result))
        results[name].append(sk.metrics.mean_squared_error(data_y[vali], result))

for name, ress in results.items():
    print("model ", name, ". mean: ", np.mean(ress), ", sd: ", np.std(ress))
