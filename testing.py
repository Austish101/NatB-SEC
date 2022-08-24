#testing ability to load weights - currently doesn't configure for loading pyshark, TBD
#in theory should run all the below and dump a CSV of all predictions in format 0.0000

import input_data
import RNN
import tensorflow as tf
import numpy as np

config = input_data.read_json("config.json")
inputs, outputs = input_data.get_inputs_and_outputs(config["training_files"])

net = RNN.RNN(config, inputs, outputs)

net.model.load_weights("./netweight.dat")

predicted_d = net.model.predict(inputs)

np.savetxt("./predicted.csv", predicted_d, fmt='%0.4f')