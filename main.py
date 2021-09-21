import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNetwork

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = ""
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",", skiprows=1)
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",", skiprows=1) 

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

NN = NeuralNetwork([20,20,20], 784, 10, ['ReLU', 'ReLU', 'ReLU', 'softmax'], 'MSE', 0.1, 0.5)


lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

for i in range(100):
    NN.train_RMSProp(train_imgs[i].transpose(), train_labels_one_hot[i])