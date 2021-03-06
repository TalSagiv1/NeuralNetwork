import numpy as np
import json
from scipy.special import softmax, expit
from Functions import ReLU, ReLU_derivative, softmax_grad, sigmoid_derivative, MSE, MSE_derivative, cross_entropy, delta_cross_entropy
import math

class NeuralNetwork:
    # initilize neural network, hidden_layers is a list of the width of each hidden layer
    # activation function is a list of activation function for every hidden layer and output layer,
    # same for derivative
    def __init__(self, hidden_layers, input_nodes, output_nodes,
     activation_functions, cost_function, learning_rate, decay_rate, weights=None, bias=None) -> None:
        self.inodes = input_nodes
        self.onodes = output_nodes
        self.hlayers = hidden_layers
        self.activation_functions = activation_functions
        self.cost_function = cost_function
        self.lr = learning_rate
        self.dr = decay_rate
        self.accumulated_grad = 0

        
        # if starting bias is not specified initilize as 1
        if bias == None:
            self.bias = [np.ones(layer) for layer in hidden_layers]
            self.bias.append(np.ones(output_nodes))
        else:
            self.bias = bias
        
        #  if starting weights are not specified initilize random weights
        # with uniform distribution (-6/sqrt(m+n),6/sqrt(m+n))
        if weights == None:
            self.weights = [np.random.uniform(-6.0/math.sqrt(input_nodes + hidden_layers[0]), 
            6.0/math.sqrt(input_nodes + hidden_layers[0]), size=(hidden_layers[0], input_nodes))]
            for i in range(len(hidden_layers) - 1):
                self.weights.append(np.random.uniform(-6.0/math.sqrt(hidden_layers[i] + hidden_layers[i+1]), 
                6.0/math.sqrt(hidden_layers[i] + hidden_layers[i+1]), size=(hidden_layers[i+1], hidden_layers[i])))
            self.weights.append(np.random.uniform(-6.0/math.sqrt(hidden_layers[-1] + output_nodes), 
                6.0/math.sqrt(hidden_layers[-1] + output_nodes), size=(output_nodes, hidden_layers[-1])))
        else:
            self.weights = weights

    # x,y are numpy arrays representing minibatch
    def calculate_gradient(self, x, y):
        dw = []  # dC/dW
        db = []  # dC/dB
        z_s, a_s = self.query(x)
        print(a_s[-1].shape)
        deltas = [None] * len(self.weights)  # Error per layer
        deltas[-1] = self.get_cost_function_derivative(self.cost_function)(y, a_s[-1]) * self.get_activation_function_derivative(
            self.activation_functions[-1])(z_s[-1])
        # Perform BackPropagation
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.get_activation_function_derivative(
                self.activation_functions[i])(z_s[i]))
        batch_size = 1 #TODO fix batch
        db = deltas
        dw = [d.dot(a_s[i].T) for i,d in enumerate(deltas)]
        # return the derivitives respect to weight matrix and biases
        return dw, db
        



    def train_RMSProp(self, x, y):
        dw, db = self.calculate_gradient(x, y)
        self.accumulated_grad = (self.dr * self.accumulated_grad) + (1 - self.dr) * np.dot(
            np.append(np.copy(dw), np.copy(db), np.append(np.copy(dw), np.copy(db))))
        coef = -self.lr / math.sqrt(math.pow(10, -6) + self.accumulated_grad)
        self.weights = [weight + coef * grad for (weight, grad) in zip(self.weights, dw)]
        self.bias = [bias + coef * grad for (bias, grad) in zip(self.bias, db)]

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
            a = self.get_activation_function(self.activation_functions[i])(z_s[-1])
            a_s.append(a)
        return (z_s, a_s)

    
    def run(self, x):
        return self.query(x)[1][-1]
            


    # saves trained weights, bias and nodes structure to .json
    def save(self, path):
        pass

    # generate new neural network from .json
    def open(path):
        pass

    def get_activation_function(self, str):
        activation_func = {
            'ReLU': np.vectorize(ReLU),
            'sigmoid': expit,
            'softmax': softmax
        }
        return activation_func[str]


    def get_activation_function_derivative(self, str):
        activation_derivative={
            'sigmoid': np.vectorize(sigmoid_derivative),
            'ReLU': np.vectorize(ReLU_derivative),
            'softmax': softmax_grad
            
        }
        return activation_derivative[str]

    def get_cost_function(self, str):
        cost_func={
            'MSE': np.vectorize(MSE),
            'cross enthropy': cross_entropy
        }
        return cost_func[str]
    
    def get_cost_function_derivative(self, str):
        cost_derivative = {
            'MSE': MSE_derivative,
            'cross enthropy': delta_cross_entropy
        }
        return cost_derivative[str]

