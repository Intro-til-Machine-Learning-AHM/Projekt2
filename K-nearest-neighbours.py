# K-nearest neighbors
# exercise 7.1.1

from matplotlib.pyplot import (figure, hold, plot, title, xlabel, ylabel,
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
os.chdir("C:\\Users\\andre\\Documents\\Machine Learning\\Projects\\Project1\\Projekt2")
os.getcwd()


# Load Matlab data file and extract variables of interest
data = pd.read_csv('data.csv')
data_vali = pd.read_csv('data_vali.csv')

X_train = data.drop("class",axis=1)
X_test = data_vali.drop("class",axis=1)

y_train = data["class"]
y_test = data_vali["class"]

attributeNames = list(X_test)
print(attributeNames)
classNames = ['Class 1','Class 2']
print(classNames)
#N, M = X.shape What is this?? - Andreas
C = len(classNames)


# Plot the training data points (color-coded) and test data points.
figure(1)
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])


# K-nearest neighbors
K=5

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=2

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist);
knclassifier.fit(X_train, y_train);
y_est = knclassifier.predict(X_test);


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
title('Synthetic data classification - KNN');

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
figure(2);
imshow(cm, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

show()

print('Ran Exercise 7.1.1')
