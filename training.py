# handle the training of the DNN

# file data must be fed into the DNN at a variable speed, it will be slower than real time to start with
import input_data
import RNN
import tensorflow as tf
import numpy as np


# get the expected output given input packets: the next error type and timestamp
def outputs_given_inputs(input_data, output_data):
    errstr_liveliness_changed = "DRIVER on_liveliness_changed"
    errstr_requested_deadline_missed = "DRIVER on_requested_deadline_missed"
    errstr_sample_lost = "DRIVER on_sample_lost"
    dumpallerrors = True

    output = []
    n = 0
    for i in range(len(input_data)):
        if output_data[n][0] < input_data[i][0]:
            n = n + 1

            # error type
            chrtemp = ""
            strtemp = ""
            # for c in pkt.DATA.data_data.split(':'):
            for c in output_data[n][1].data_data.split(':'):
                try:
                    chrtemp = chr(int(c, 16))
                    strtemp = strtemp + chrtemp
                    # print(chr(int(c,16)), end='')
                except ValueError:
                    pass
            if errstr_liveliness_changed in strtemp or errstr_requested_deadline_missed in strtemp or errstr_sample_lost in strtemp:
                if dumpallerrors:
                    print("At time: " + output_data[n][0] + " : ", end='')
                    print(strtemp, end='')
                if errstr_liveliness_changed in strtemp:
                    error_type = [output_data[n][0], 1, 0, 0]
                elif errstr_requested_deadline_missed in strtemp:
                    error_type = [output_data[n][0], 0, 1, 0]
                elif errstr_sample_lost in strtemp:
                    error_type = [output_data[n][0], 0, 0, 1]

        output.append(error_type)
    return output


# split a set of data into two, given a percentage
def split_data(split_percentage, data):
    data_len = len(data)
    split_point = round(data_len * (split_percentage / 100))
    split1 = data[:split_point]
    split2 = data[split_point:]
    return split1, split2


def get_sd_mean(data):
    sd = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    return sd, mean


def standard_data(data, sd, mean):
    std_data = data

    for p in range(0, data.len()):
        for d in range(0, data[0].len()):
            std_data[p][d] = (data[p][d] - mean[d]) / sd[d]

    return std_data


def inverse_standard(data, sd, mean):
    real_data = data

    for p in range(0, data.len()):
        for d in range(0, data[0].len()):
            real_data[p][d] = (data[p][d] * sd[d]) + mean[d]

    return real_data


config = input_data.read_json("config.json")
inputs, outputs = input_data.get_inputs_and_outputs(config["training_files"])

expected_outputs = outputs_given_inputs(inputs, outputs)
training_in, testing_in = split_data(config["split"], inputs)
training_out, testing_out = split_data(config["split"], expected_outputs)

input_sd, input_mean = get_sd_mean(training_in)
output_sd, output_mean = get_sd_mean(training_out)
training_in_std = standard_data(training_in, input_sd, input_mean)
training_out_std = standard_data(training_out, output_sd, output_mean)
testing_in_std = standard_data(testing_in, input_sd, input_mean)
testing_out_std = standard_data(testing_out, output_sd, output_mean)


net = RNN.RNN(config, training_in_std, training_out_std)

i = 0
tf.random.set_seed(5)
random_error = tf.random.uniform(shape=[], minval=0, maxval=2)  # random error type
random_timestamp = tf.random.uniform(shape=[], minval=inputs[0][0], maxval=inputs[len(inputs)-1][0])  # random timestamp
if random_error == 0:
    previous_output = [random_timestamp, 1, 0, 0]
elif random_error == 1:
    random_output = [random_timestamp, 0, 1, 0]
else:
    random_output = [random_timestamp, 0, 0, 1]

for pkt in inputs:
    pkt.extend(previous_output)
    actual_output = net.produce_output(pkt)
    net.update(pkt, expected_outputs[i], actual_output)
    previous_output = actual_output
    i = i + 1
    # TODO print the output and related reward (accuracy)

# TODO second loop for testing data

# TODO save the weights and the sd/means

net.model.save_weights("./netweight.dat")