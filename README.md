# Federated-Learning

This is a simple reproduction to partly implement and simulate the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629 "FedAvg") by two GPUs.<br />
We use ZeroMQ which is an open-source universal messaging library to exchange messages between two processes. One is master and the other is device for simulation. The master is responsible for aggregating the delta weights and the device trains its data on different GPUs which represents different devices.

## Example
Open two terminals to simulate the master and device.

    python fed_device.py

Run the device to wait the master instructions (creating devices for training).

    python fed_master.py

Run the master to distribute the training job to the device.

If you want to adjust more detailed settings, you can modify the json file (i.e., data/detailed_settings.json).