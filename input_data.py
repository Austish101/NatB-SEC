# handles the reading and formatting of data for the DNN
# https://github.com/KimiNewt/pyshark/
import pyshark
from scapy.utils import PcapReader
import json


# get inputs from rtps file, get expected output from label file
def get_inputs_and_output(filename):
    rtps_capture = pyshark.FileCapture(filename + ".RTPS.pcap")
    input_list = []
    for pkt in rtps_capture:
        # extract pertinent data e.g. actual payload, from packet into input_list
        if pkt.highest_layer == "RTPS":
            input_list.append([pkt.ip.src_host, pkt.ip.dst_host, pkt.sniff_timestamp, pkt.rtps])
    # for packet in PcapReader(filename + ".RTPS.pcap"):
    #     # extract pertinent data e.g. actual payload, from packet into input_list
    #     input_list.append(packet)

    lbl_capture = pyshark.FileCapture(filename + ".LABEL.pcap")
    output_list = []
    for pkt in lbl_capture:
        # extract pertinent data e.g. error data, from packet into output_list
        if pkt.highest_layer == "DATA":
            output_list.append([pkt.DATA, pkt.sniff_timestamp])
    return input_list, output_list


def read_file(filename):
    data = "todo_read"
    return data


def read_json(filename):
    try:
        with open(filename, "rb") as fileData:
            json_data = json.load(fileData)
            return json_data
    except OSError:
        print("ERROR: Can't read file, ensure it exists")
        quit()

