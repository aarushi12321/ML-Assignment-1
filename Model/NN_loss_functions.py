import numpy as np

def mean_squared_error(y, y_pred):
    return np.mean(np.power((y - y_pred),2))

def mean_squared_error_derivative(y, y_pred):
    return 2*(y_pred-y)/y.size

def accuracy(y, y_pred):
    assert len(y) == len(y_pred), "The lengths of actual and predicted labels must match."
    correct_predictions = sum(y == y_pred)
    accuracy = correct_predictions / len(y)
    
    return accuracy