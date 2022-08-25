
import RNN
import tensorflow as tf
import numpy as np
import pyshark
import input_data #only need this one to get a small file to setup the array sizes - might be able to get rid of later

time_out = 20 #set this far higher, or basically infinitely high to run forever...

config = input_data.read_json("config.json")
#create a file that has basically nothing but two or three samples, so it loads quickly to initialise the arrays.
#alternatively, generate a blank array of the right lengths.
inputs_temp, outputs_temp = input_data.get_inputs_and_outputs(config["training_files"])
team = config("team")
ifname = config("networkname") #Wi-Fi 2 just what I'm testing with....

net = RNN.RNN(config, inputs_temp, outputs_temp)

def pkt_callback(pkt):
    input_line = input_data.get_line(pkt)  # new function we're going to need, will work better. Needs to get just one input line using pkt data
    predict_d = net.model.predict([input_line], verbose=0)
    # believe we need to have an output category for "everything is fine" from some prior testing
    # if thats the case then check against predict_d[0][0] (the OK category) for detection?
    #if predict_d[0][n] > threshold or predict_d[0][0]:
    #   nestAPI.on_liveliness_changed(team, alive_count, not_alive_count)
    # if predict_d[0][n] > threshold or predict_d[0][0]:
    #   nestAPI.on_samplelost(team, lost_reason)
    # if predict_d[0][n] > threshold or predict_d[0][0]:
    #   nestAPI.on_liveliness_changed(team, missed_total_count)

net.model.load_weights("./netweight.dat")

capture = pyshark.LiveCapture(interface=ifname)

capture.apply_on_packets(print_callback, timeout=time_out)





