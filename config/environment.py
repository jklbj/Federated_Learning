# Use zeromq to exchange messages
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
    
    gpu_num = str(device_num%num_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    with K.tf.device('/device:GPU:%s'%gpu_num):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        

# ZeroMQ connect to specific port (master)
def zmq_connect():
    """
    Use zmq to exchange messages between devices and master.
    """
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5566")

    return socket

# ZeroMQ bind to the specific port (device) 
def zmq_bind():
    """
    Use zmq to exchange messages between devices and master.
    """
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5566")

    return socket