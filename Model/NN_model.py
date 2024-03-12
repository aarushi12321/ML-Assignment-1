from Model.NN_loss_functions import *

class NNModel():
    def __init__(self, args):
        self.args = args
        self.layers = []
        self.loss, self.loss_derivative = globals()[args.loss_function], globals()[args.loss_function + '_derivative']

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        y_pred = []
        for i in range(len(input_data)):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_pass(output)
            
            if output < 0.5 :
                output = 0
            else :
                output = 1
            y_pred.append(output)

        return y_pred
    
    def fit(self, x_train, y_train, verbose):
        # iterate over the number of epochs
        error_per_epoch = []
        accuracy_per_epoch = []

        for e in range(self.args.epochs):
            error = 0
            # begin an iteration of forward and backward pass
            for i in range(len(x_train)):
                output = x_train[i]

                # forward pass
                for layer in self.layers:
                    output = layer.forward_pass(output)
                
                error += self.loss(y_train[i], output)

                # backward pass
                error_layer = self.loss_derivative(y_train[i], output)
                for layer in reversed(self.layers):
                    error_layer = layer.backward_pass(error_layer)
                
            # calculate mean error
            error = error / len(x_train)
            y_pred = self.predict(x_train)
            train_accuracy_current = accuracy(y_train, y_pred)
            error_per_epoch.append(error)
            accuracy_per_epoch.append(train_accuracy_current)

            if verbose == 1:
                print('epoch %d/%d \t error=%f \t accuracy=%f' % (e+1, self.args.epochs, error, train_accuracy_current))
            
        return error_per_epoch, accuracy_per_epoch

