import numpy as np

def broadcast_to_device(model_x, model_m):
    """
    Broadcast to every device (e.g., model_m parameters to all devices)
    :param the client model to receive the weight for this round
    :param the master model to send the weight for this round
    """
    for layer in range(len(model_m.layers)):
        model_x.layers[layer].set_weights(model_m.layers[layer].get_weights())

def caculate_delta(model_x, model_m):
    """
    Calculate delta weight on each device
    :param the model to be caculated
    :param the model to store the delta weight results
    """
    for layer in range(len(model_m.layers)):
        model_x.layers[layer].set_weights(
            np.subtract(model_x.layers[layer].get_weights(),
                model_m.layers[layer].get_weights()))
        
def aggregate_add(model_x, model_0):
    """
    Aggregate all delta weight to the temp device (e.g., device 0) (Addition)
    :param the model to send its trained weight
    :param the model (e.g., device 0) to store others' weights temporarily
    """
    for layer in range(len(model_0.layers)):
        model_0.layers[layer].set_weights(
            np.add(model_x.layers[layer].get_weights(),
                model_0.layers[layer].get_weights()))

def aggregate_division_return(model_0, model_m, num_device):
    """
    Return total delta weight to center device
    :param the temp model (device 0) which just stored all the clients' weights
    :param master model to aggregate all the results from device 0
    :param number of devices which is divisor
    """
    for layer in range(len(model_m.layers)):
        aggregate_to_master(aggregate_division(model_0, num_device, layer), model_m, layer)
    
def aggregate_division(model_0, num_device, layer):
    """
    Aggregate all delta weight on device 0 (Division)
    :param the temp model (device 0) which just stored all the clients' weights
    :param number of devices which is divisor
    :param what layer is now
    :return the divided result in this layer
    """
    return np.divide(model_0.layers[layer].get_weights(), num_device)

def aggregate_to_master(model_0_result, model_m, layer):
    """
    Return total delta weight to center device
    :param the division result from the temp model (device 0)
    :param master model to aggregate all the results from device 0
    :param what layer is now
    """
    model_m.layers[layer].set_weights(np.add(model_m.layers[layer].get_weights(), model_0_result))