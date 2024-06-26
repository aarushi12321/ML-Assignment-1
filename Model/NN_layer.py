import numpy as np
from Model.NN_activation import *
from Model.NN_optimizer import *

class NNLayer():
    def __init__(self, args, input_size, output_size):
        self.args = args
        self.input = None
        self.output = None
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)
        self.args_temp = vars(args)
        self.optimizer = None
        optimizer_class = globals().get(self.args_temp['optimizer'])
        if optimizer_class:
            self.optimizer = optimizer_class(self.args)
    
    def forward_pass(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_pass(self, output_error):
        input_T = self.input.reshape(-1, 1) if self.input.ndim == 1 else self.input.T
        change_err_by_input = np.dot(output_error, self.weights.T)
        change_err_by_weights = np.dot(input_T, output_error)
        change_err_by_bias = np.sum(output_error, axis=0, keepdims=True)

        # updating the weights and biases
        if self.optimizer == None:
            self.weights = self.weights - self.args.learning_rate * (change_err_by_weights)
            self.bias = self.bias - self.args.learning_rate * (change_err_by_bias)
        else:
            self.optimizer.update(self.weights, change_err_by_weights)
            self.optimizer.update(self.bias, change_err_by_bias)

        return change_err_by_input
    
class ActivationLayer():
    def __init__(self, args):
        self.args = args
        self.activation_function = globals()[args.activation_function]
        self.activation_function_derivative = globals()[args.activation_function + '_derivative']
        self.input, self.output = None, None

    def forward_pass(self, input_data):
        self.input = input_data
        self.output = self.activation_function(self.input)
        return self.output

    def backward_pass(self, output_error):
        return self.activation_function_derivative(self.input)* output_error


