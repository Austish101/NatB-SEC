#testing ability to load weights - currently doesn't configure for loading pyshark, TBD
#in theory should run all the below and dump a CSV of all predictions in format 0.0000

import input_data
import RNN
import tensorflow as tf
import numpy as np
import sys
import LSTM

config = input_data.read_json("config.json")

#copied from training.py
def standard_data(data, sd, mean, kind="all"):
    std_data = data

    if data.shape.__len__() == 1:
        for d in range(0, data.shape[0]):
            calc = (data[d] - mean[d]) / sd[d]
            std_data[d] = float(calc)
        return std_data

    for p in range(0, data.shape[0]):
        if kind == "non-error":
            calc = (data[p][0] - mean[0]) / sd[0]
            std_data[p][0] = float(calc)
        else:
            for d in range(0, data[0].shape[0]):
                calc = (data[p][d] - mean[d]) / sd[d]
                std_data[p][d] = float(calc)

    return std_data

#copied from training.py
def get_sd_mean(data):
    sd = np.std(data, axis=0, dtype=float)
    mean = np.mean(data, axis=0, dtype=float)
    return sd, mean

#need to load an input and output array to configure the network.
try:
    training_in_all = np.loadtxt('training_in.txt', dtype=float)
    training_out_all = np.loadtxt('training_out.txt', dtype=float)
    testing_in_all = np.loadtxt('testing_in.txt', dtype=float)
    testing_out_all = np.loadtxt('testing_out.txt', dtype=float)
except:
    print("Training files are currently required to initialise the neural network.")
    sys.exit()

input_sd, input_mean = get_sd_mean(training_in_all)
output_sd, output_mean = get_sd_mean(training_out_all)
training_in_std = standard_data(training_in_all, input_sd, input_mean)
training_out_std = standard_data(training_out_all, output_sd, output_mean, "non-error")
testing_in_std = standard_data(testing_in_all, input_sd, input_mean)
testing_out_std = standard_data(testing_out_all, output_sd, output_mean, "non-error")

#scores = []
net = LSTM.SplitLSTM(config, training_in_std, training_out_std)
net.load_models("", "") #the params aren't currently used...

time_score, error_score, combined_score = net.predict(testing_in_std, testing_out_std)
print("Time Score:", time_score, "\nError Score:", error_score, "\nCombined:", combined_score)
#scores.append([time_score, error_score])
#print("Time Score:", time_score, "\nError Score:", error_score)

#need to be able to get an output that states the exact data, time + error type. doesn't seem to be one at the moment?

#predicted_d = net.model.predict(inputs)

#np.savetxt("./predicted.csv", predicted_d, fmt='%0.4f')