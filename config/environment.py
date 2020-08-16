#!/usr/bin/env python
# coding: utf-8

# Zmq
import multiprocessing as mp
import zmq
from keras import backend as K
import tensorflow as tf
import os

# Use specific GPU
def gpu_decision(num_gpu=2 ,device_num=0):
    """
    Decide which GPU to compute.
    :param the number of GPUs
    :param the number of devices
    """
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num%num_gpu)
    with K.tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
# Bind zmq
def zmq_bind():
    """
    Use zmq to exchange messages between devices and master.
    """
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5566")

    return socket