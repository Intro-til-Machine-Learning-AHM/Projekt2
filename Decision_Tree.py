# Tree-code stolen from exercise 5.1.6
import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
import os
os.chdir("C:\\Users\\andre\\Documents\\Machine Learning\\Projects\\Project1\\Projekt2")
os.getcwd()
# requires data
# Load datafiles
data = pd.read_csv('data.csv')
data_vali = pd.read_csv('data_vali.csv')

# Assign data to train and test
X_train = data.drop("class",axis=1)
X_test = data_vali.drop("class",axis=1)
y_train = data["class"]
y_test = data_vali["class"]

#Assign attributeNames and stuff
attributeNames = list(X_train)
classNames = ['Class 1','Class 2']

#Convert to matrix form
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()

# Compute values of N, M and C.
N = len(y_train)
M = len(attributeNames)
C = len(classNames)

# Fit regression tree classifier, Gini split criterion, pruning enabled
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=50)
dtc = dtc.fit(X_train,y_train)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini_fat_indians.dot', feature_names=attributeNames)
#graph = graphviz.Source(out)
#graph.render('dtree_render',view=True)
#src=graphviz.Source.from_file('tree_gini_Wine_data.gvz')
#src.render('.\\tree_gini_Wine_data.gvz', view=True)
print('Ran Exercise 5.1.6')
