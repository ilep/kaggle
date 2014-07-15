# -*- coding: utf-8 -*-
"""
Created on Mon Jul 07 19:46:37 2014

@author: ivan lepoutre
"""

import pandas as pd
import numpy as np
import os
import platform
# Standard scientific Python imports

if platform.system() == 'Windows':
    if os.path.exists(os.getcwd() +'\\train.csv'):
        path = os.getcwd() +'\\train.csv' # on met dans path le chemin du training set, si ce chemin existe

data = pd.read_csv(path,delimiter=',')
n = data.shape[0]  # n est le nombre de ligne
p = data.shape[1]  # p le nombre de colonne = nombre de pixels dans les images. Attention la premiere colonne est le label de l'image 


y = data['label'] # labels 
X = data.iloc[:,1:p] # features


#-------------------------------- 0 INTRODUCTION ---------------------------------------
import pylab as pl

# 0/1 plot the 9 first digit 
nine_first_digit = X.iloc[0:9,:] # on recupere les 9 premiers digits
for i in range(nine_first_digit.shape[0]): # et on les trace pour voir si çà correspond bien aux labels  
    pl.subplot(3, 3, i + 1) # on se place sur le ieme element de notre grille de 3*3 plots
    digit = nine_first_digit.iloc[i,:].values.reshape((28,28)) # la ligne representant les pixels est transformee en une matrice numpy 28*28
    pl.imshow(digit, cmap=pl.cm.gray_r, interpolation='nearest') # on trace cette matrice de pixels qui representent l'image de resolution 28*28

#-------------------------------- I DATA ANALYSIS ---------------------------------------
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA 
from pylab import *

# I/ Outlier detection 

# I/1 covariance between features

# I/2 representtaion of the importance of features
svc = SVC(kernel="linear", C=1) 
rfe = RFE(estimator=svc, n_features_to_select=1, step=1) # recursive feature elimination en utilisant SVC
Xbis = X.iloc[0:1000,:]
ybis = y.iloc[0:1000]
rfe.fit(Xbis,ybis) # RFE est fitte sur 1000 digits pour etre plus rapide ds un premier temps
ranking = rfe.ranking_.reshape((28,28)) # le classement des pixels est mis sous forme de matrice numpy
pl.matshow(ranking) # on peut s'apercevoir de l'importance des pixels du milieu par rapport a ceux des bordures qui ne contiennet aucune info
pl.colorbar()
pl.title("Ranking of pixels with RFE")
pl.show()

# I/3 feature elimination
features_to_keep = np.asarray(np.where(rfe.ranking_ <= 350)).reshape(350) # on garde arbitrairement les 350 pixels le splus explicatifs du label au sens de RFE
Xbis = X.iloc[:,features_to_keep]

# I/4 feature selection using PCA
pca = PCA(n_components=81) 
X_new = pca.fit_transform(X)
print(pca.explained_variance_ratio_) # sum is equal to 1
plot(pca.explained_variance_ratio_)

nine_first_digit_new = X_new[0:9,:] # X_new.iloc[0:9,:]
for i in range(nine_first_digit_new.shape[0]):  
    pl.subplot(3, 3, i + 1)
    digit = nine_first_digit_new[i,:].reshape((9,9))
    pl.imshow(digit, cmap=pl.cm.gray_r, interpolation='nearest')




#-------------------------------- II PREDICTION -----------------------------------------
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# II/1 prediction using support vector machine classification 
classifier = svm.SVC(gamma = 0.0001)
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.33, random_state=42)
print type(X_train)
classifier.fit(X_train, y_train) # We learn the digits on the training set 
y_hat = classifier.predict(X_test) # Now predict the value of the digit on the test set


# II/2 prediction using support k neighborhood classification 
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train) 




#-------------------------------- III DEEP LEARNING --------------------------------------
from theano import *

# III/1 LeNet5 neural network (https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/convolutional_mlp.py)


#-------------------------------- IV TESTING ---------------------------------------------


print(__doc__)

# Standard scientific Python imports
import pylab as pl

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits,
# let's have a look at the first 3 images, stored in the `images`
# attribute of the dataset. If we were working from image files, we
# could load them using pylab.imread. For these images know which
# digit they represent: it is given in the 'target' of the dataset.
for index, (image, label) in enumerate(zip(digits.images, digits.target)[:4]):
    pl.subplot(2, 4, index + 1)
    pl.axis('off')
    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    pl.title('Training: %i' % label)

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

for index, (image, prediction) in enumerate(
        zip(digits.images[n_samples / 2:], predicted)[:4]):
    pl.subplot(2, 4, index + 5)
    pl.axis('off')
    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    pl.title('Prediction: %i' % prediction)

pl.show()






