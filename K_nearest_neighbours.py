# K-nearest neighbors
# script stolen from exercise 7.1.1

from matplotlib.pyplot import (figure, hold, plot, title, xlabel, ylabel,
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
from sklearn import model_selection

os.chdir("C:\\Users\\andre\\Documents\\Machine Learning\\Projects\\Project1\\Projekt2")
os.getcwd()

data = pd.read_csv('data.csv')

def KNN(x,y):
    # Load datafiles
    X_train = x
    y_train = y

    # Assign data to train and test

    #Convert to matrix form
    #X_train = X_train.as_matrix()
    #y_train = y_train.as_matrix()

    attributeNames = list(X_train)
    classNames = ['Class 1','Class 2']
    C = len(classNames)

    # Plot the training data points (color-coded) and test data points.
    #figure(1)
    #styles = ['.b', '.r', '.g', '.y']
    #for c in range(C):
        #class_mask = (y_train==c)
        #plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])


    # K-nearest neighbors
    K=15 # See output plot for why 15 is the best K

    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=2

    # Fit classifier and classify the test points
    knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_train);


    # Plot the classfication results
    #styles = ['ob', 'or', 'og', 'oy']
    #for c in range(C):
        #class_mask = (y_est==c)
        #plot(X_train[class_mask,0], X_train[class_mask,1], styles[c], markersize=10)
        #plot(X_train[class_mask,0], X_train[class_mask,1], 'kx', markersize=8)
    #title('Synthetic data classification - KNN');

    # Compute and plot confusion matrix
    #cm = confusion_matrix(y_train, y_est);
    #accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    #figure(2);
    #imshow(cm, cmap='binary', interpolation='None');
    #colorbar()
    #xticks(range(C)); yticks(range(C));
    #xlabel('Predicted class'); ylabel('Actual class');
    #title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

    #show()

    print('Ran Exercise 7.1.1')

    # Validation

    # Compute values of N, M and C.
    N = len(y_train)
    M = len(attributeNames)
    C = len(classNames)

    # Maximum number of neighbors
    L=50

    CV = model_selection.LeaveOneOut()
    errors = np.zeros((N,L))
    i=0
    for train_index, test_index in CV.split(X_train, y_train):
        #print('Crossvalidation fold: {0}/{1}'.format(i+1,N))
        #print(train_index)
        #print(test_index)

        # extract training and test set for current CV fold
        X_train1 = X_train[train_index,:]
        y_train1 = y_train[train_index]
        X_test1 = X_train[test_index,:]
        y_test1 = y_train[test_index]

        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train1, y_train1);
            y_est = knclassifier.predict(X_test1);
            errors[i,l-1] = np.sum(y_est[0]!=y_test1[0])

        i+=1

    # Plot the classification error rate
    #figure()
    #plot(100*sum(errors,0)/N)
    #xlabel('Number of neighbors')
    #ylabel('Classification error rate (%)')
    #show()
    index_min = np.argmin(sum(errors))
    best_k = index_min + 1
    knclassifier = KNeighborsClassifier(n_neighbors=best_k)
    knclassifier.fit(X_train, y_train)

    return (knclassifier.predict, best_k)

#print(KNN(X_train1,y_train1))
