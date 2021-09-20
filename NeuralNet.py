import numpy as np
import json

class NeuralNetwork:
    #initilize neural network, hidden_layers is a list of the width of each
    #hidden layer
    def __init__(self, hidden_layers, input_nodes, output_nodes,
     activation_function, activation_function_derivative, weights=None) -> None:
        #if starting weights are not specified initilize random weights
        #with distribution NOTCHOSENYET
        #TODO choose distribution and implement
        if weights == None:
            weights = [None for i in range(len(hidden_layers) + 1)]
            for i in range(len(hidden_layers)):
                pass
        
    def train_RMSProp(self, minibatch):
        pass

    def train_RMSProp_momentum(self, minibatch):
        pass

    def train_AdaGrad(self, minibatch):
        pass

    def query(self, inputs):
        pass

    #saves trained weights and nodes structure to .json
    def save(self, path):
        pass

    #generate new neural network from .json
    def open(path, activation_function,
     activation_function_derivative):
     pass