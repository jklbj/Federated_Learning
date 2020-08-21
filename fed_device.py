# Modified fedavg for our input
from __future__ import absolute_import, division, print_function, unicode_literals

# Other import
import sys
import os
import tensorflow as tf
from data.preprocessing import preprocessing_for_training, separate_and_preprocess_for_simple_fed, evaluate_with_new_model
from data.read_data import read_data, read_setting
from data.data_utils import load_cifar10_data, train_test_label_to_categorical
from model.model import init_model, record_history, training_once, print_result_for_fed
from model.operation import broadcast_to_device, caculate_delta, aggregate_add, aggregate_division_return 
from config.environment import gpu_decision, zmq_bind

from math import floor
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# Use zeromq to exchange message between master and devices
import multiprocessing as mp
import zmq

_, epo = 0, 0

def step_decay(epoch):
    epoch = _ + epo
    # initial_lrate = 1.0 # no longer needed
    drop = 0.99
    epochs_drop = 1.0
    
    lrate = 0.2 * pow(drop, floor((1+epoch)/epochs_drop))
    return lrate

def model_training(r, device, training_info, weight_source):
    # Load CIFAR-10 data
    train_images, train_labels, test_images, test_labels = load_cifar10_data()

    # Adjust parameters of the model
    callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
    # Define the method for preprocessing
    augment = ImageDataGenerator(preprocessing_function=preprocessing_for_training)
    
    # Decide which GPU to use
    gpu_decision(training_info["num_gpu"] ,device)
            
    model_m = tf.keras.models.load_model(weight_source+'model_m.h5')

    # Define our model
    if(r == 0):
        locals()['model_{}'.format(device)] = init_model()
    else:
        locals()['model_{}'.format(device)] = tf.keras.models.load_model(weight_source+'model_m.h5')
    
    # Local training on each device
    for epo in range(training_info["num_local_epoch"]):
        train_new_image, train_new_label = separate_and_preprocess_for_simple_fed(device, train_images, train_labels, training_info["num_device"])
        history_temp = training_once(locals()['model_{}'.format(device)], train_new_image, train_new_label, training_info, augment, callback)

    # Calculate delta weight on each device
    caculate_delta(locals()['model_{}'.format(device)], model_m)
        
    # Save delta weight in each model weight(.h5) file 
    try:
        if(r == 0):
            locals()['model_{}'.format(device)].save(weight_source+'model_{}.h5'.format(device))
        else:
            locals()['model_{}'.format(device)].save_weights(weight_source+'model_{}.h5'.format(device))
    except:
        print("Can't not save the model!")

def main(argv):
    # bind zmq socket port
    socket = zmq_bind()

    # Read detailed settings from json file
    detailed_setting = read_setting()
    training_info = detailed_setting["training_info"]
    weight_source = detailed_setting["file_source"]["devices_weight_folder"]

    while True:
        while True:
            # Recieve message from master
            message = socket.recv()
            ms = str(message, encoding = "utf-8").split(":")
            r = int(ms[0])
            device = int(ms[1])
            print("(Received) round: %s, device: %s" % (message, device))
            
            # Start process for the device
            locals()['p_{}'.format(device)] = mp.Process(target=model_training, args=(r, device, training_info, weight_source))
            locals()['p_{}'.format(device)].start()
            
            # Send message back to master
            # Tell master that the process (device) starts local training
            socket.send(b"ok")
            if(device == (training_info['num_device']-1)): break
        
        # Wait all processes done the jobs (local training)
        for device in range(training_info['num_device']):
            locals()['p_{}'.format(device)].join()

        # Tell master that the 'r' round finishs
        message = socket.recv()
        socket.send(b"finish")
        if(r == (training_info['num_round']-1)): break

if __name__ == '__main__':
    main(sys.argv)