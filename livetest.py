# To Do
#
# - Add Syncronising Routine (as wont be the same time on hosts as this client)
# - May need additional pipe to return sync offset from thread routine, likely bidirectional
# - Add check for current time in the time thread

import RNN
import support
import tensorflow as tf
import numpy as np
import pyshark
import input_data #only need this one to get a small file to setup the array sizes - might be able to get rid of later
import json

import output_data

firstpkt = True #this is to get around pyshark being super buggy on first call
from multiprocessing import Process, Pipe
from multiprocessing import Pool
import multiprocessing
import time

#if you have a working directory problem this makes finding it way easier...
print('Reading Config Json: ', end='')
config = input_data.read_json("config.json")
print('Done.')

#create a file that has basically nothing but two or three samples, so it loads quickly to initialise the arrays.
#alternatively, generate a blank array of the right lengths.
#inputs_temp, outputs_temp = input_data.get_inputs_and_outputs(config["training_files"])
team = config["team"]
port = config["port"]
api_url = config["api_url"]
ifname = config["networkname"] #Wi-Fi 2 just what I'm testing with....
time_out = config["timeout"] #set this far higher, or basically infinitely high to run forever...

pktarray = np.zeros((3, 9), dtype='float') #edit this to be dynamic, get 3+length of params list from config
pipe_recv, pipe_send = Pipe()
syncpipe_recv, syncpipe_send = Pipe()

#a time offset for syncronising and a check for if it has been initialised yet or not.
#use the first packet to do this...
#syncoffset = 0
#syncinit = False

testvar = False

def shiftpktarray(pktarray_in, pkt):
    newpktarray = np.roll(pktarray_in, 1, axis=0)
    newpktarray[0] = pkt
    return newpktarray

#=================================================================================================================
# The time management thread and the packet detection thread
#=================================================================================================================

def threads(net, threadno, pipe, syncpipe):
    if threadno == 0:
        print("Thread 0 Start")
        runthread = True
        syncinit = False
        syncoffset = 0
        sent_already_if_true = False #enforces pulses for errors to occur only ONCE.
        alive_count = 0
        not_alive_count = 0
        missed_total_count = 0
        lost_reason = "DDS_LOST_BY_WRITER"
        while runthread:
            # Sync is because we can't be certain the network is time synced to the PI
            # This therefore resolves that issue.
            if not syncinit:
                if syncpipe.poll():
                    syncoffset = syncpipe.recv()
                    syncinit = True
            if pipe.poll():
                pipearray = pipe.recv()
                #print("Packet Detected")
                #==================================================
                # SEND TO REST API IF ERROR PREDICTED
                #==================================================
                if (pipearray[0] + syncoffset < time.time()): #is the predicted time less than actual time? then error has happened.
                    if sent_already_if_true == False: #ensures only one message per error prediction
                        sent_already_if_true = True
                        if pipearray[1] > pipearray[2] and pipearray[1] > pipearray[3]:
                            #insert any change for alive_count or not_alive count here if desired
                            #seems to generally be once and 1 or 0, suggest don't bother.
                            output_data.REST_on_liveliness_changed(
                                port, api_url, team, alive_count, not_alive_count)
                        elif pipearray[2] > pipearray[1] and pipearray[2] > pipearray[3]:
                            missed_total_count = missed_total_count + 1
                            output_data.REST_on_requested_deadline_missed(
                                port, api_url, team, missed_total_count)
                        else:
                            #insert any change for lost_reason here if desired
                            #all examples in provided data is DDS_LOST_BY_WRITER which I've set as default, suggest don't bother.
                            output_data.REST_on_sample_lost(
                                port, api_url, team, lost_reason)
                else:
                    sent_already_if_true = False

            #    if temp != "end":
            #    else:
            #        print("Killing Time Monitor")
            #        runthread = False
            #global testvar
            #if testvar == True:
            #    testvar = False
            #    print("Packet Detected")
    else:
        print("Thread 1 Start")
        syncoffset = float(0)
        syncinit = False
        _, output_sd_mean = net.get_sd_mean()
        time_sd = output_sd_mean[0][0]
        time_mean = output_sd_mean[1][0]
        error_sd = output_sd_mean[0][1:]
        error_mean = output_sd_mean[1][1:]
        #def checknetwork_thread():
        capture = pyshark.LiveCapture(interface=ifname)
        for pkt in capture.sniff_continuously():
            syncinit, syncoffset = pkt_callback(pkt, pipe, syncpipe, syncinit, syncoffset, time_sd, time_mean)
        #try:
        #    capture.apply_on_packets(pkt_callback, timeout=time_out)
        #except:
        #    print("Capture Time Completed or error, restart program if more required.")
        pipe.send("end")

#Note that pktarray may have problems - will need to pass this through all defs down to this level.
def pkt_callback(net, pkt, pipe_send, syncpipe_send, syncinit, syncoffset, time_sd, time_mean):
    #global testvar
    #testvar = True
    #global pipe_send
    if syncinit == False:
        syncoffset = float(pkt.sniff_timestamp) - time.time()
        syncpipe_send(syncoffset)
    #pipe_send.send(pkt.sniff_timestamp)
    if pkt.highest_layer == "RTPS":
        input_line = input_data.get_input_line(pkt, config["rtps_selection"])
        global pktarray
        #global firstpkt
        #if firstpkt == True:
        #    firstpkt = False
        #    return
        pktarray = shiftpktarray(pktarray, input_line)
        #predict_d = net.model.predict(input_line, verbose=0) #if this fails put "input_line" into [ ] brackets.
        predict_error = net.error_model.predict(pktarray)
        predict_time = net.time_model.predict(pktarray)
        predict_time = (predict_time * time_sd) + time_mean #reverse the standardisation
        pipearray = np.concatenate((predict_time, predict_error))
        pipe_send.send(pipearray)
    return syncinit, syncoffset

if __name__ == '__main__':
    # need to load an input and output array to configure the network.
    try:
        training_in_all = np.loadtxt('training_in.txt', dtype=float)
        training_out_all = np.loadtxt('training_out.txt', dtype=float)
        testing_in_all = np.loadtxt('testing_in.txt', dtype=float)
        testing_out_all = np.loadtxt('testing_out.txt', dtype=float)
    except:
        print("A training file is required to initialise the neural network.")
        # sys.exit()
        quit()

    input_sd, input_mean = support.get_sd_mean(training_in_all)
    output_sd, output_mean = support.get_sd_mean(training_out_all)
    training_in_std = support.standard_data(training_in_all, input_sd, input_mean)
    training_out_std = support.standard_data(training_out_all, output_sd, output_mean, "non-error")
    testing_in_std = support.standard_data(testing_in_all, input_sd, input_mean)
    testing_out_std = support.standard_data(testing_out_all, output_sd, output_mean, "non-error")

    # scores = []
    net = RNN.Split(config, training_in_std, training_out_std, input_mean, input_sd, output_mean, output_sd)
    net.load_models("", "")  # the params aren't currently used...
    # net = RNN.RNN(config, inputs_temp, outputs_temp)

    print("Capture Started")
    print("number of CPU detected: ", multiprocessing.cpu_count())
    with Pool(2) as p:
        p.starmap(threads, [(0, 0, pipe_recv, syncpipe_recv), (net, 1, pipe_send, syncpipe_send)])
        #= Process(target=threads, args=(i,))
        #jobs.append(p)
        #p.start()
    #with Pool(processes=2) as pool:
    #    pool.apply_async(func=threads, args=[0, 1])
    #    while True:
    #        pass
    #x = Process(target=checknetwork_thread(), args=(0,), daemon=True)
    #y = Process(target=checktimes_thread(), args=(0,), daemon=True)
    #y.start()
    #x.start()
    #x.join()
    #y.join()
    #while x.is_alive():
    #    pass
    #y.close()






