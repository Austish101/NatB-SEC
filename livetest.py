
import RNN
import tensorflow as tf
import numpy as np
import pyshark
import input_data #only need this one to get a small file to setup the array sizes - might be able to get rid of later
import json

config = input_data.read_json("config.json")
#create a file that has basically nothing but two or three samples, so it loads quickly to initialise the arrays.
#alternatively, generate a blank array of the right lengths.
inputs_temp, outputs_temp = input_data.get_inputs_and_outputs(config["training_files"])
team = config["team"]
ifname = config["networkname"] #Wi-Fi 2 just what I'm testing with....
time_out = config["timeout"] #set this far higher, or basically infinitely high to run forever...

net = RNN.RNN(config, inputs_temp, outputs_temp)

def pkt_callback(pkt):
    input_line = input_data.get_input_line(pkt, config["rtps_selection"])
    predict_d = net.model.predict(input_line, verbose=0) #if this fails put "input_line" into [ ] brackets.
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
          predict_d[0][0] + ", " +
          predict_d[0][1] + ", " +
          predict_d[0][2] + ", " +
          predict_d[0][3])

net.model.load_weights("./netweight.dat")

capture = pyshark.LiveCapture(interface=ifname)

capture.apply_on_packets(print_callback, timeout=time_out)





