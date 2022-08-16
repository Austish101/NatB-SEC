# handles the reading and formatting of data for the DNN
# https://github.com/KimiNewt/pyshark/
import pyshark
import json


# get inputs from rtps file, get expected output from label file
# TODO data MUST be flat (only a 1d list, not nested), can't do 'pkt.rtps' or 'pkt.DATA' as below
# TODO data should be normalised, not sure how yet, preferably between 0-1 for all inputs and outputs,
#       TODO ^^ above also has to work when reading off the wire
def get_inputs_and_outputs(filename):
    rtps_capture = pyshark.FileCapture(filename + ".RTPS.pcap")
    input_list = []
    _list = []
    print("Loading Input Files")
    for pkt in rtps_capture:
        # extract pertinent data e.g. actual payload, from packet into input_list
        # TODO include other non-rtps packets, but must be in the same format as the rtps packets
        if pkt.highest_layer == "RTPS":
            input_list.append([float(pkt.sniff_timestamp)])#, pkt.ip.src_host, pkt.ip.dst_host, pkt.rtps])
            #commenting off a lot of the content to get this to run - most of this is STRING values.
            #will ammend this to individual values later.

    lbl_capture = pyshark.FileCapture(filename + ".LABEL.pcap")
    output_list = []
    for pkt in lbl_capture:
        # extract pertinent data e.g. error data, from packet into output_list
        if pkt.highest_layer == "DATA":
            output_list.append([float(pkt.sniff_timestamp), pkt.DATA])

    print("Finished Loading Input Files")
    return input_list, output_list


def read_json(filename):
    try:
        with open(filename, "rb") as fileData:
            json_data = json.load(fileData)
            return json_data
    except OSError:
        print("ERROR: Can't read file, ensure it exists")
        quit()


# TODO use pyshark to read packets from interface (likely eno1), format the same as the file version
def read_from_wire(interface):
    return interface
