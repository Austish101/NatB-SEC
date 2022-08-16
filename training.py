# handle the training of the DNN

# file data must be fed into the DNN at a variable speed, it will be slower than real time to start with
import input_data
import RNN
import tensorflow as tf


# get the expected output given input packets: the next error type and timestamp
def outputs_given_inputs(input_data, output_data):
    errstr_liveliness_changed = "DRIVER on_liveliness_changed"
    errstr_requested_deadline_missed = "DRIVER on_requested_deadline_missed"
    errstr_sample_lost = "DRIVER on_sample_lost"
    dumpallerrors = True

    output = []
    n = 0
    for i in range(len(input_data)):
        error_type = [0, 0, 0, 0]
        if output_data[n][0] < input_data[i][0]:
            # error type
            chrtemp = ""
            strtemp = ""
            for c in output_data[n][1].data_data.split(':'):
                try:
                    chrtemp = chr(int(c, 16))
                    strtemp = strtemp + chrtemp
                    # print(chr(int(c,16)), end='')
                except ValueError:
                    pass
            if errstr_liveliness_changed in strtemp or errstr_requested_deadline_missed in strtemp or errstr_sample_lost in strtemp:
                if dumpallerrors:
                    print("t: " + str(output_data[n][0]) + " n: " + str(n) + " i: " + str(i) + " : ", end='')
                    print(strtemp, end='')
                if errstr_liveliness_changed in strtemp:
                    error_type = [output_data[n][0], 1, 0, 0]
                elif errstr_requested_deadline_missed in strtemp:
                    error_type = [output_data[n][0], 0, 1, 0]
                elif errstr_sample_lost in strtemp:
                    error_type = [output_data[n][0], 0, 0, 1]

            else:
                #print("## Warning : No Error identified where an error should have been detected.")
                #print("## t: " + output_data[n][0] + " n: " + str(n) + " i: " + str(i) + " : ", end='')

                error_type = [output_data[n][0], 0, 0, 0]
            if n < (len(output_data)-1):
                n = n + 1

        output.append(error_type)
    return output


# split a set of data into two, given a percentage
def split_data(split_percentage, data):
    data_len = len(data)
    split_point = round(data_len * (int(split_percentage) / 100))
    split1 = data[:split_point]
    split2 = data[split_point:]
    return split1, split2


config = input_data.read_json("config.json")
inputs, outputs = input_data.get_inputs_and_outputs(config["training_files"])

expected_outputs = outputs_given_inputs(inputs, outputs)
training_in, testing_in = split_data(config["split"], inputs)
training_out, testing_out = split_data(config["split"], expected_outputs)

net = RNN.RNN(config, training_in, training_out)

i = 0
tf.random.set_seed(5)
previous_error = tf.random.uniform(shape=[], minval=0, maxval=2)  # random error type
previous_timestamp = tf.random.uniform(shape=[], minval=inputs[0][0], maxval=inputs[len(inputs)-1][0])  # random timestamp
for pkt in inputs:
    pkt.append(previous_error)
    pkt.append(previous_timestamp)
    actual_output = net.produce_output(pkt)
    previous_error = actual_output[0]
    previous_timestamp = actual_output[1]
    net.update(pkt, expected_outputs[i], actual_output)
    i = i + 1
