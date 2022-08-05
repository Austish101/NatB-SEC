# handles the reading and formatting of data for the DNN
# https://github.com/KimiNewt/pyshark/
import pyshark
import json
import data_store

errstr_liveliness_changed = "DRIVER on_liveliness_changed"
errstr_requested_deadline_missed = "DRIVER on_requested_deadline_missed"
errstr_sample_lost = "DRIVER on_sample_lost"

dumpallerrors = True

# get inputs from rtps file, get expected output from label file
# todo data MUST be flat (only a 1d list, not nested), can't do 'pkt.rtps' or 'pkt.DATA' as below
# todo data should be normalised, not sure how yet, preferably between 0-1
def get_inputs_and_outputs(filename):
    print("Extracting data from file : " + filename + ".RTPS.pcap" + "\nPlease wait: ", end='')
    rtps_capture = pyshark.FileCapture(filename + ".RTPS.pcap")
    input_list = []
    _list = []
    prev_pkt_sniff_time = float(rtps_capture[0].sniff_timestamp) #this is so first relative value always zero
    input_len = 0 #needed for generating the output array
    for pkt in rtps_capture:
        # extract pertinent data e.g. actual payload, from packet into input_list
        # do not exclude non-RTPS packets. These can give hints to failures about to occur.
        # e.g. check timing of on_sample_lost DDS_LOST_BY_WRITER, as these occur when
        # a IPv4 fragmented protocol occurs in the RTPS input file. It is likely I beleive
        # that this is more down to the size of such messages (1516) than the protocol though.
        #
        # /removed this line/ if pkt.highest_layer == "RTPS":
        input_len = input_len + 1
        #input_list.append([float(pkt.sniff_timestamp) - prev_pkt_sniff_time,
        #                   pkt.ip.src_host, pkt.ip.dst_host, pkt.rtps])
        data_store.input_d.append([float(pkt.sniff_timestamp),
                                    float(pkt.sniff_timestamp) - prev_pkt_sniff_time,
                                    pkt.length,
                                    pkt.captured_length])
        prev_pkt_sniff_time = float(pkt.sniff_timestamp)
        #make sure user is aware this hasn't crashed, because this is very slow
        if input_len%200 == 0:
            print('.', end='')
    print("") #new line, as above loop doesn't make one.
    print("Input Array Length : " + str(input_len))

    lbl_capture = pyshark.FileCapture(filename + ".LABEL.pcap")
    #temp output list is - [timestamp, liveliness_changed_error, requested_deadline_error, sample_lost_error]
    temp_output_list = []
    for pkt in lbl_capture:
        # extract pertinent data e.g. error data, from packet into output_list
        if pkt.highest_layer == "DATA":

            chrtemp = ""
            strtemp = ""
            for c in pkt.DATA.data_data.split(':'):
                try:
                    chrtemp = chr(int(c,16))
                    strtemp = strtemp + chrtemp
                    #print(chr(int(c,16)), end='')
                except ValueError:
                    pass
            if errstr_liveliness_changed in strtemp or errstr_requested_deadline_missed in strtemp or errstr_sample_lost in strtemp:
                if dumpallerrors == True:
                    print("At time: " + pkt.sniff_timestamp + " : ", end='')
                    print(strtemp, end='')
                if errstr_liveliness_changed in strtemp:
                    temp_output_list.append([float(pkt.sniff_timestamp), 1, 0, 0])
                elif errstr_requested_deadline_missed in strtemp:
                    temp_output_list.append([float(pkt.sniff_timestamp), 0, 1, 0])
                elif errstr_sample_lost in strtemp:
                    temp_output_list.append([float(pkt.sniff_timestamp), 0, 0, 1])
                else:
                    temp_output_list.append([float(pkt.sniff_timestamp), 0, 0, 0])

            #temp_output_list.append([pkt.sniff_timestamp, pkt.DATA])

    #setup the data_store.output as a inputlength x 4 2d array of zeroes
    data_store.output_d = [[9, 9, 9, 9]] * input_len

    print("input_d  len : " + str(len(data_store.input_d)))
    print("output_d len : " + str(len(data_store.output_d)))

    data_store.temp_array = temp_output_list

    print("Writing output_d data...")

    i_temp = 0 #position in the temp_output_list
    for i in range(0, input_len):
        if i_temp < len(temp_output_list):
            if data_store.input_d[i][0] == temp_output_list[i_temp][0]:
                print("= t: " + str(data_store.input_d[i][0]) + " i: " + str(i) +
                      " i_temp: " + str(i_temp) + " :: ",end="")
                data_store.output_d[i][0] = 0
                data_store.output_d[i][1] = temp_output_list[i_temp][1]
                data_store.output_d[i][2] = temp_output_list[i_temp][2]
                data_store.output_d[i][3] = temp_output_list[i_temp][3]
                print(data_store.output_d[i])
                i_temp = i_temp + 1
            elif data_store.input_d[i][0] > temp_output_list[i_temp][0]:
                print("> t: " + str(data_store.input_d[i-1][0]) + " i: " + str(i-1) +
                      " i_temp: " + str(i_temp) + " :: ",end="")
                data_store.output_d[(i-1)][0] = temp_output_list[i_temp][0] - data_store.input_d[i-1][0]
                data_store.output_d[(i-1)][1] = temp_output_list[i_temp][1]
                data_store.output_d[(i-1)][2] = temp_output_list[i_temp][2]
                data_store.output_d[(i-1)][3] = temp_output_list[i_temp][3]
                data_store.output_d[i] = [0,0,0,0]
                print(data_store.output_d[i-1])
                i_temp = i_temp + 1
            else:
                data_store.output_d[i] = [0,0,0,0]

    #this should NOT be used- just here for testing at the moment. need to replace with
    #code for rewriting output list
    #data_store.output_d = temp_output_list
    
    #this function doesn't need a return, the data_store does this for us.
    #return input_list, temp_output_list

def get_next():
    input_list  = data_store.input_d[data_store.i]
    output_list = data_store.output_d[data_store.i]
    data_store.i = data_store.i + 1
    return input_list, output_list


def read_json(filename):
    try:
        with open(filename, "rb") as fileData:
            json_data = json.load(fileData)
            return json_data
    except OSError:
        print("ERROR: Can't read file, ensure it exists")
        quit()


# todo use pyshark to read packets from interface (likely eno1), format the same as the file version
def read_from_wire(interface):
    return interface
