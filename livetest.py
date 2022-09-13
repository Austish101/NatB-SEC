
import RNN
import tensorflow as tf
import numpy as np
import pyshark
import input_data #only need this one to get a small file to setup the array sizes - might be able to get rid of later
import LSTM
import json
firstpkt = True #this is to get around pyshark being super buggy on first call

config = input_data.read_json("config.json")
#create a file that has basically nothing but two or three samples, so it loads quickly to initialise the arrays.
#alternatively, generate a blank array of the right lengths.
#inputs_temp, outputs_temp = input_data.get_inputs_and_outputs(config["training_files"])
team = config["team"]
ifname = config["networkname"] #Wi-Fi 2 just what I'm testing with....
time_out = config["timeout"] #set this far higher, or basically infinitely high to run forever...

pktarray = np.zeros((3, 9), dtype='float') #edit this to be dynamic, get 3+length of params list from config

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

def shiftpktarray(self, pktarray_in, pkt):
    newpktarray = np.roll(pktarray_in, 1, axis=0)
    newpktarray[0] = pkt
    return newpktarray

#need to load an input and output array to configure the network.
try:
    training_in_all = np.loadtxt('training_in.txt', dtype=float)
    training_out_all = np.loadtxt('training_out.txt', dtype=float)
    testing_in_all = np.loadtxt('testing_in.txt', dtype=float)
    testing_out_all = np.loadtxt('testing_out.txt', dtype=float)
except:
    print("A training file is required to initialise the neural network.")
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

#net = RNN.RNN(config, inputs_temp, outputs_temp)

def pkt_callback(pkt):
    if pkt.highest_layer == "RTPS":
        input_line = input_data.get_input_line(pkt, config["rtps_selection"])
        global pktarray
        #global firstpkt
        #if firstpkt == True:
        #    firstpkt = False
        #    return
        pktarray = (pktarray, input_line)
        #predict_d = net.model.predict(input_line, verbose=0) #if this fails put "input_line" into [ ] brackets.
        predict_error = net.error_model.predict(pktarray)
        predict_time = net.time_model.predict(pktarray)
        # believe we need to have an output category for "everything is fine" from some prior testing
        # if thats the case then check against predict_d[0][0] (the OK category) for detection?
        #if predict_d[0][n] > threshold or predict_d[0][0]:
        #   nestAPI.on_liveliness_changed(team, alive_count, not_alive_count)
        # if predict_d[0][n] > threshold or predict_d[0][0]:
        #   nestAPI.on_samplelost(team, lost_reason)
        # if predict_d[0][n] > threshold or predict_d[0][0]:
        #   nestAPI.on_liveliness_changed(team, missed_total_count)
        # print any diagnostics here... e.g:
        print("T: " + pkt.sniff_timestamp + " P: " +
              predict_error[0][0] + ", " +
              predict_error[0][1] + ", " +
              predict_error[0][2] + ", " +
              predict_error[0][3] +
              predict_time[0])

capture = pyshark.LiveCapture(interface=ifname)

try:
    capture.apply_on_packets(pkt_callback, timeout=time_out)
except:
    print("Capture Time Completed or error, restart program if more required.")





