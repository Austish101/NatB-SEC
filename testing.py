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
import pyshark


def shiftpktarray(pktarray_in, pkt):
    #newpktarray = np.roll(pktarray_in, 1, axis=0)
    #newpktarray[0] = pkt[0]
    newpktarray = np.roll(pktarray_in, 9, axis=1)
    newpktarray[0][0:9] = pkt[0]
    return newpktarray

#=================================================================================================================
# The time management thread and the packet detection thread
#=================================================================================================================


def threads(net, threadno, pipe, syncpipe, tag):
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
                        print('Error Detected @:' + str(pipearray[0]))
                        if pipearray[1] > pipearray[2] and pipearray[1] > pipearray[3]:
                            #insert any change for alive_count or not_alive count here if desired
                            #seems to generally be once and 1 or 0, suggest don't bother.
                            output_data.post_on_liveliness_changed(
                                port, api_url, team, alive_count, not_alive_count)
                        elif pipearray[2] > pipearray[1] and pipearray[2] > pipearray[3]:
                            missed_total_count = missed_total_count + 1
                            output_data.post_on_requested_deadline_missed(
                                port, api_url, team, missed_total_count)
                        else:
                            #insert any change for lost_reason here if desired
                            #all examples in provided data is DDS_LOST_BY_WRITER which I've set as default, suggest don't bother.
                            output_data.post_on_sample_lost(
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
        #_, output_sd_mean = net.get_sd_mean(tag)
        #time_sd = output_sd_mean[0][0]
        #time_mean = output_sd_mean[1][0]
        #error_sd = output_sd_mean[0][1:]
        #error_mean = output_sd_mean[1][1:]
        #def checknetwork_thread():
        filepath = 'Dataset/newDataset/'
        filename = 'SECH.IncrementalDelayEth1'
        rtpsPath = filepath + filename + '.RTPS.pcap'
        print(rtpsPath)
        rtps_capture = pyshark.FileCapture(rtpsPath)
        syncoffsetT1 = 0
        syncinitT1 = False
        pktarray = np.zeros((1, 225), dtype='float')
        pktarrayindex = 0
        for pkt in rtps_capture:
            syncinitT1, syncoffsetT1, pktarray, pktarrayindex = pkt_callback(net, pkt, pipe, syncpipe, syncinitT1, syncoffsetT1, pktarray, pktarrayindex)



#Note that pktarray may have problems - will need to pass this through all defs down to this level.
def pkt_callback(net, pkt, pipe_send, syncpipe_send, syncinit, syncoffset, pktarray, pktarrayindex): # , time_sd, time_mean):
    #global testvar
    #testvar = True
    #global pipe_send
    if syncinit == False:
        syncoffset = float(pkt.sniff_timestamp) - time.time()
        syncpipe_send.send(syncoffset)
    else: #this else is for offline testing version only, delete this and content for livetest
        while float(pkt.sniff_timestamp) - syncoffset > time.time():
            pass
    #pipe_send.send(pkt.sniff_timestamp)
    #print("packet detected")
    if pkt.highest_layer == "RTPS":
        input_line = input_data.get_input_line(pkt, config["rtps_selection"])
        #pipe_send.send(input_line)
        pktarray[0][(pktarrayindex * 9) : ((pktarrayindex + 1) * 9)] = input_line[0]
        pktarrayindex = pktarrayindex + 1
        if pktarrayindex == 15:
            pktarrayindex = 0
            pipe_send.send(pktarray)
    return syncinit, syncoffset, pktarray, pktarrayindex


if __name__ == "__main__":

    # Main Variables
    # if you have a working directory problem this makes finding it way easier...
    print('Reading Config Json: ', end='')
    config = input_data.read_json("config.json")
    print('Done.')
    tag = '1x100'
    team = config["team"]
    port = str(config["port"])
    api_url = config["api_url"]
    ifname = config["networkname"]  # Wi-Fi 2 just what I'm testing with....
    time_out = config["timeout"]  # set this far higher, or basically infinitely high to run forever...
    #pktarray = np.zeros((3, 9), dtype='float')  # edit this to be dynamic, get 3+length of params list from config
    pktarray = np.zeros((1,225), dtype='float')
    pipe_recv, pipe_send = Pipe()
    syncpipe_recv, syncpipe_send = Pipe()
    pktpipe_recv, pktpipe_send = Pipe()
    testvar = False

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
    print('debugline1')
    net = RNN.Split(config, training_in_std, training_out_std, input_mean, input_sd, output_mean, output_sd)
    print('debugline2')
    net.load_models("error_model" + tag, "time_model" + tag)  # the params aren't currently used...
    # net = RNN.RNN(config, inputs_temp, outputs_temp)

    print("Capture Started")
    print("number of CPU detected: ", multiprocessing.cpu_count())
    #with Pool(processes=2) as p:
    #    p.starmap(threads, [(net, 1, pipe_send, syncpipe_send, tag), (0, 0, pipe_recv, syncpipe_recv, tag)])

    p0 = multiprocessing.Process(target=threads, args=(0, 0, pipe_recv, syncpipe_recv, tag))
    p1 = multiprocessing.Process(target=threads, args=(net, 1, pktpipe_send, syncpipe_send, tag))
    p1.start()
    p0.start()

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

    _, output_sd_mean = net.get_sd_mean(tag)
    time_sd = output_sd_mean[0][0]
    time_mean = output_sd_mean[1][0]

    print("Thread 2 go")
    while True:
        while not pktpipe_recv.poll():
            pass
        pktarray = pktpipe_recv.recv()
        #predict_d = net.model.predict(input_line, verbose=0) #if this fails put "input_line" into [ ] brackets.
        predict_error = net.error_model.predict(pktarray)
        predict_time = net.time_model.predict(pktarray)
        predict_time = (predict_time * time_sd) + time_mean #reverse the standardisation
        pipearray = np.concatenate((predict_time, predict_error), axis=1)
        print("Predict_Success!!! Predict_Success!!! Predict_Success!!! Predict_Success!!! ")
        pipe_send.send(pipearray[0])





