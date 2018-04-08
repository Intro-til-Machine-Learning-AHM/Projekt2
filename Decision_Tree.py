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
out = tree.export_graphviz(dtc, out_file='tree_gini_fat_indians.dot', feature_names=attributeNames)

print('Ran Exercise 5.1.6')

tc = np.arange(2, 20, 1)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

k=0
for train_index, test_index in CV.split(X_train):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train1, y_train1 = X_train[train_index,:], y_train[train_index]
    X_test1, y_test1 = X_train[test_index,:], y_train[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train1,y_train1.ravel())
        y_est_test = dtc.predict(X_test1)
        y_est_train = dtc.predict(X_train1)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = sum(np.abs(y_est_test - y_test1)) / float(len(y_est_test))
        misclass_rate_train = sum(np.abs(y_est_train - y_train1)) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    k+=1


f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])

show()

print('Ran Exercise 6.1.2')

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV.split(X):

    # extract training and test set for current CV fold
    X_train1 = X_train[train_index,:]
    y_train1 = y_train[train_index]
    X_test1 = X_train[test_index,:]
    y_test1 = y_train[test_index]
    internal_cross_validation = 10

    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train1-y_train1.mean()).sum()/y_train1.shape[0]
    Error_test_nofeatures[k] = np.square(y_test1-y_test1.mean()).sum()/y_test1.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train1, y_train1)
    Error_train[k] = np.square(y_train1-m.predict(X_train1)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    #textout = 'verbose';
    textout = '';
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)

    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]

        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')

        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1


# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')


# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

f=2 # cross-validation fold to inspect
ff=Features[:,f-1].nonzero()[0]
if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
else:
    m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)

    y_est= m.predict(X[:,ff])
    residual=y-y_est

    figure(k+1, figsize=(12,6))
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       subplot(2,np.ceil(len(ff)/2.0),i+1)
       plot(X[:,ff[i]],residual,'.')
       xlabel(attributeNames[ff[i]])
       ylabel('residual error')


show()

print('Ran Exercise 6.2.1')
