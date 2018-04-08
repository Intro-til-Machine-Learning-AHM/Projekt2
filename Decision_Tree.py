# Tree-code stolen from exercise 5.1.6
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import model_selection
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, boxplot, legend
from scipy.io import loadmat
import sklearn.linear_model as lm
from toolbox_02450 import feature_selector_lr, bmplot
import graphviz
import os

def Tree(x,y):
    # Assign data to train and test
    X_train = x
    y_train = y

    #Assign attributeNames and stuff
    attributeNames = list(X_train)
    classNames = ['Class 1','Class 2']

    # Compute values of N, M and C.
    N = len(y_train)
    M = len(attributeNames)
    C = len(classNames)

    # Fit regression tree classifier, Gini split criterion, pruning enabled
    dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=50)
    dtc = dtc.fit(X_train,y_train)

    # Export tree graph for visualization purposes:
    out = tree.export_graphviz(dtc, out_file='tree_gini_fat_indians.dot')#, feature_names=attributeNames)


    tc = np.arange(2, 20, 1)

    # K-fold crossvalidation
    #K = 10
    #CV = model_selection.KFold(n_splits=K,shuffle=True)
    CV = model_selection.LeaveOneOut()

    # Initialize variable
    # Error_train = np.empty((len(tc),K))
    # Error_test = np.empty((len(tc),K))

    #k=0
    #for train_index, test_index in CV.split(X_train):
    #    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

        # extract training and test set for current CV fold
    #    X_train1, y_train1 = X_train[train_index,:], y_train[train_index]
    #    X_test1, y_test1 = X_train[test_index,:], y_train[test_index]

    #    for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
    #        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
    #        dtc = dtc.fit(X_train1,y_train1.ravel())
    #        y_est_test = dtc.predict(X_test1)
    #        y_est_train = dtc.predict(X_train1)
            # Evaluate misclassification rate over train/test data (in this CV fold)
    #        misclass_rate_test = sum(np.abs(y_est_test - y_test1)) / float(len(y_est_test))
    #        misclass_rate_train = sum(np.abs(y_est_train - y_train1)) / float(len(y_est_train))
    #        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    #    k+=1
    # Compute values of N, M and C.
    N = len(y_train)
    M = len(attributeNames)
    C = len(classNames)

    #Initialize variable
    Error_train = np.empty((len(tc),len(y_train)))
    Error_test = np.empty((len(tc),len(y_train)))

    CV = model_selection.LeaveOneOut()
    errors = np.zeros((N,len(tc)))
    k = 0
    for train_index, test_index in CV.split(X_train, y_train):
        #print('Crossvalidation fold: {0}/{1}'.format(k+1,N))

        #extract training and test set for current CV fold
        X_train1, y_train1 = X_train[train_index], y_train[train_index]
        X_test1, y_test1 = X_train[test_index], y_train[test_index]

        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for i, t in enumerate(tc):
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train1,y_train1.ravel())
            y_est_test = dtc.predict(X_test1)
            y_est_train = dtc.predict(X_train1)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = sum(np.abs(y_est_test - y_test1)) / float(len(y_est_test))
            misclass_rate_train = sum(np.abs(y_est_train - y_train1)) / float(len(y_est_train))
            Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
        k+=1

    #f = figure()
    #boxplot(Error_test.T)
    #xlabel('Model complexity (max tree depth)')
    #ylabel('Test error, Leave one out)'.format(N))

    #f = figure()
    #plot(tc, Error_train.mean(1))
    #plot(tc, Error_test.mean(1))
    #xlabel('Model complexity (max tree depth)')
    #ylabel('Error (misclassification rate, Leave one out)'.format(N))
    #legend(['Error_train','Error_test'])

    #show()

    index_min = np.argmin(Error_test.mean(1))
    best_depth = tc[index_min]
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=best_depth)
    dtc = dtc.fit(X_train,y_train.ravel())

    return (dtc.predict,best_depth)
