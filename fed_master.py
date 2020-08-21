# Modified fedavg for our input
from __future__ import absolute_import, division, print_function, unicode_literals

# Other import
import sys
import os
import time
import tensorflow as tf
from data.preprocessing import preprocessing_for_training, separate_and_preprocess_for_simple_fed, evaluate_with_new_model
from data.read_data import read_data, read_setting
from data.data_utils import load_cifar10_data, train_test_label_to_categorical
from model.model import init_model, record_history, training_once, print_result_for_fed
from model.operation import broadcast_to_device, caculate_delta, aggregate_add, aggregate_division_return 
from config.environment import gpu_decision, zmq_bind, zmq_connect

from math import floor
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# Use zeromq to exchange message between master and devices
import multiprocessing as mp
import tensorflow as tf
import zmq

def main(argv):
    # Connect zeromq socket port
    socket = zmq_connect()
    # Decide gpu
    gpu_decision()
    # Read detailed settings from json file
    detailed_setting = read_setting()
    # Load CIFAR-10 data
    train_images, train_labels, test_images, test_labels = load_cifar10_data()
    # Transfer train and test label to be categorical
    train_label, test_label = train_test_label_to_categorical(train_labels, test_labels)

    training_info = detailed_setting["training_info"]
    weight_source = detailed_setting["file_source"]["devices_weight_folder"]

    # Define our model
    is_master = True 
    model_m, history_total = init_model(is_master)

    # Measure the start training time
    start = time.time()

    for _ in range(training_info["num_round"]):
        print("\n" + "\033[1m" + "Round: " + str(_))
        start_training = time.time()
        
        for device in range(training_info["num_device"]):
            print("\033[0m" + "Device:", device, "model_" + str(device))
            message = "{}:{}".format(_, device)
            socket.send(bytes(message, encoding = "utf8"))
            response = socket.recv()
            print("response: %s" % response)  
        
        # Wait for all devices to complete training
        while True:
            socket.send(b"aggregate")
            message = socket.recv()
            print("Received: %s" % message)
            if(message == b"finish"): break
                
        if(_ == 0):
            # Define and initialize an estimator model
            for device in range(training_info["num_device"]):
                globals()['model_{}'.format(device)] = tf.keras.models.load_model(weight_source+'model_{}.h5'.format(device))
        else:
            for device in range(training_info["num_device"]):
                globals()['model_{}'.format(device)].load_weights(weight_source+'model_{}.h5'.format(device))                                 

        # Aggregate all delta weight on device 0 (Addition)
        for device in range(1, training_info["num_device"]):
            aggregate_add(globals()['model_{}'.format(device)], model_0)

        # Aggregate all delta weight on device 0 (Division) and return total delta weight to center device
        aggregate_division_return(model_0, model_m, training_info["num_device"])

        # Evaluate and save with new weight
        history_temp = evaluate_with_new_model(_, training_info, model_m, test_images, test_label)
        model_m.save(weight_source+'model_m.h5')

        # Record each round accuracy
        record_history(history_temp, history_total)
        
    print("training time: ", (time.time() - start))
    print_result_for_fed(history_total)

if __name__ == '__main__':
    main(sys.argv)