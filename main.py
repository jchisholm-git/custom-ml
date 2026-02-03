from preprocessing.pca import PCA
from models.knn import KNN
from models.linear_regression import LinearRegression

from preprocessing.standard import StandardScaler
import utils.data_helpers as dh

from keras.datasets import boston_housing
import numpy as np


(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()
train_X_flat = dh.flatten_features(train_X)
test_X_flat = dh.flatten_features(test_X)

ss = StandardScaler()
train_x_stand = ss.fit_transform(train_X_flat)
test_x_stand = ss.transform(test_X_flat)

"""pca = PCA(.99, "randomized")
train_x_reduced = pca.fit_transform(train_x_stand)
test_x_reduced = pca.transform(test_x_stand)

knn = KNN(5, "euclidean")
knn.fit(train_x_reduced, train_Y)
print(knn.evaluate(test_x_reduced, test_Y))"""



lr = LinearRegression('normal', .01, 'constant', lr=.01, batch_size=32, epochs=100)
lr.fit(train_x_stand, train_Y)
print(lr.parameters)
