import numpy as np
import json

class NeuralNetwork:
    # initilize neural network, hidden_layers is a list of the width of each hidden layer
    # activation function is a list of activation function for every hidden layer and output layer,
    # same for derivative
    def __init__(self, hidden_layers, input_nodes, output_nodes,
     activation_functions, activation_functions_derivative, cost_function,
      cost_function_derivative, weights=None, bias=None) -> None:
        self.inodes = input_nodes
        self.onodes = output_nodes
        self.hlayers = hidden_layers
        self.activation_functions = activation_functions
        self.activation_functions_derivative = activation_functions_derivative
        self.cost_function = cost_function
        self.cost_function_derivative = cost_function_derivative
        
        # if starting bias is not specified initilize as 1
        if bias == None:
            self.bias = []
            for i in range(len(hidden_layers)):
                self.bias.append(np.ones(hidden_layers[i]))
            self.bias.append(np.ones(output_nodes))
        else:
            self.bias = bias
        
        #  if starting weights are not specified initilize random weights
        # with distribution NOTCHOSENYET
        # TODO choose distribution and implement
        if weights == None:
            self.weights = []
            for i in range(len(hidden_layers) + 1):
                pass
        else:
            self.weights = weights
    # x,y are numpy arrays representing minibatch
    def calculate_gradient(self, x, y):
        '''partial_derivative = [None for i in range(self.inodes)]
        partial_derivative = [[None for i in range(self.hidden_layer)] for self.hidden_layer in self.hidden_layers]
        # nested function to calculate the partial derivatives by backwards propagation
        def calculate_gradiante_recursive(self, i, j):
            if i == len(self.hlayers):
                pass
            elif i == len(self.hlayers - 1):
                pass
            else:
                for node in range(self.hlayers[i+1]):
                    pass
                '''
        dw = []  # dC/dW
        db = []  # dC/dB
        z_s, a_s = self.query(x)
        deltas = [None] * len(self.weights)  # Error per layer
        deltas[-1] = self.cost_function_derivative(y-a_s[-1]) * self.activation_functions_derivative[-1](z_s[-1])
        # Perform BackPropagation
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.activation_functions_derivative[i](z_s[i]))        
            batch_size = y.shape[1]
            db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
            dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]
            # return the derivitives respect to weight matrix and biases
            return dw, db
        



    def train_RMSProp(self, minibatch):
        pass

    def train_RMSProp_momentum(self, minibatch):
        pass

    def train_AdaGrad(self, minibatch):
        pass

    # Feed forwards
    def query(self, inputs):
        a = np.copy(inputs)
        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            z_s.append(self.weights[i].dot(a) + self.bias[i])
            a = self.activation_functions[i](z_s[-1])
            a_s.append(a)
            return (z_s, a_s)
            


    # saves trained weights, bias and nodes structure to .json
    def save(self, path):
        pass

    # generate new neural network from .json
    def open(path, activation_function, activation_function_derivative,
    cost_function, cost_function_derivative,):
        pass