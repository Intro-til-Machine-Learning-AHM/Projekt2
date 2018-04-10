from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from scipy import stats
import os
os.chdir("C:\\Users\\andre\\Documents\\Machine Learning\\02450Toolbox_Python\\Scripts")
os.getcwd()
Error_logreg1 =np.array([0.78,0.75,0.71,0.78,0.77]) #knn
Error_dectree1 = np.array([0.67,0.67,0.67,0.67,0.67]) #dectree / dummy
[tstatistic, pvalue] = stats.ttest_ind(Error_logreg1,Error_dectree1)
print(pvalue)
print(tstatistic)
K = 5
Error_logreg = np.empty((K,1))
Error_dectree = np.empty((K,1))
for i in range(0,5):
    Error_logreg[i] = Error_logreg1[i]
    Error_dectree[i] = Error_dectree1[i]
z = (Error_logreg - Error_dectree)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')
else:
    print('Classifiers are significantly different.')

# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_logreg, Error_dectree),axis=1))
xlabel('KNN   vs.   Tree')
ylabel('Cross-validation accuracy [%]')

show()

print('Ran Exercise 6.3.1')
