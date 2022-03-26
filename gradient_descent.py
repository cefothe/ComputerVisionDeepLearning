from copy import deepcopy
import numpy as np

def feed_forward(inputs,outputs, weights):
    pre_hidden = np.dot(inputs.weights[0])+ weights[1]
    hidden = 1/(1+np.exp(-pre_hidden))
    pred_out = np.dot(hidden,weights[2]) + weights[3]
    mean_squared_error = np.mean(np.square(pred_out-outputs))
    return mean_squared_error

def update_weight(inputs, outputs, weights, lr):
    original_weights = deepcopy(weights)
    temp_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)
    original_loss = feed_forward(inputs, outputs, original_weights)
    for i, layer in enumerate(original_weights):
        for index, weights in np.ndenumerate(layer):
            temp_weights = deepcopy(weights)
            temp_weights[i][index] += 0.0001
            _loss_plus = feed_forward(inputs, outputs, temp_weights)
            grad = (_loss_plus - original_loss) / (0.0001)
            updated_weights[i][index]-= grad*lr
    return updated_weights, original_loss
