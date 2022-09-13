# handles the reading and formatting of data for the DNN
# https://github.com/KimiNewt/pyshark/
import pyshark
import json
import numpy as np


def get_rtps_data(rtps, selection):
    selected_data = []
    if selection[0] != "all":
        for attr in selection:
            attr_val = rtps.get_field_value(attr)
            if attr_val == None:
                selected_data.append(float(2))
            else:
                selected_data.append(float(attr_val))
    return selected_data


# get inputs from rtps file, get expected output from label file
# done using extend? - data MUST be flat (only a 1d list, not nested), can't do 'pkt.rtps' or 'pkt.DATA' as below
# done but standardisation instead - data should be normalised, not sure how yet, preferably between 0-1 for all inputs and outputs,
#       TODO ^^ above also has to work when reading off the wire
def get_inputs_and_outputs(filepath, filename, rtps_selection):
    rtps_capture = pyshark.FileCapture(filepath + filename + ".RTPS.pcap")
    input_list = []
    for pkt in rtps_capture:
        # extract pertinent data e.g. actual payload, from packet into input_list
        # TODO include other non-rtps packets, but must be in the same format as the rtps packets
        if pkt.highest_layer == "RTPS":
            # input_list.append([pkt.sniff_timestamp, pkt.ip.src_host, pkt.ip.dst_host, pkt.rtps])
            # TODO ips should probably include 2 and 3 to cover multicast etc
            input_line = get_input_line(pkt, rtps_selection)
            input_list = input_list + input_line
    rtps_capture.close()

    lbl_capture = pyshark.FileCapture(filepath + filename + ".LABEL.pcap")
    output_list = []
    for pkt in lbl_capture:
        # extract pertinent data e.g. error data, from packet into output_list
        if pkt.highest_layer == "DATA":
            output_list.append([float(pkt.sniff_timestamp), pkt.DATA.data_data])
    lbl_capture.close()

    return np.array(input_list), np.array(output_list)


def read_json(filename):
    try:
        with open(filename, "rb") as fileData:
            json_data = json.load(fileData)
            return json_data
    except OSError:
        print("ERROR: Can't read file, ensure it exists")
        quit()

def get_input_line(pkt, rtps_selection):
    src_ip = pkt.ip.src_host.split(".")
    dst_ip = pkt.ip.dst_host.split(".")
    input_line = []
    input_line.append([float(pkt.sniff_timestamp), float(src_ip[3]), float(dst_ip[3])])
    input_line[-1].extend(get_rtps_data(pkt.rtps, rtps_selection))
    return input_line

# TODO use pyshark to read packets from interface (likely eno1), format the same as the file version
#def read_from_wire(interface):
    #return interface
