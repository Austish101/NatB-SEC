# handle the training of the DNN

# file data must be fed into the DNN at a variable speed, it will be slower than real time to start with
import input_data
import RNN
import tensorflow as tf
import numpy as np


# get the expected output given input packets: the next error type and timestamp
def outputs_given_inputs(input_data, output_data, split):
    errstr_liveliness_changed = "DRIVER on_liveliness_changed"
    errstr_requested_deadline_missed = "DRIVER on_requested_deadline_missed"
    errstr_sample_lost = "DRIVER on_sample_lost"
    dumpallerrors = True


    train_outputs = []
    test_outputs = []
    train_inputs = []
    test_inputs = []
    n = 0
    for i in range(len(input_data)):
        if (float(output_data[n][0]) < input_data[i][0]) or (n == 0):
            error_found = False
            # find the next occuring error
            while not error_found:
                n = n + 1

                # cut off the tail of input data where there is no error
                if n == output_data.shape[0]:
                    break

                # 'split' the training and testing data
                error_rand = np.random.randint(0, 100)
                if error_rand <= split:
                    data_set = "train"
                else:
                    data_set = "test"

                # set the error type to the next occurring error
                chrtemp = ""
                strtemp = ""
                # for c in pkt.DATA.data_data.split(':'):
                for c in output_data[n][1].split(':'):
                    try:
                        chrtemp = chr(int(c, 16))
                        strtemp = strtemp + chrtemp
                        # print(chr(int(c,16)), end='')
                    except ValueError:
                        pass
                if (errstr_liveliness_changed in strtemp) or (errstr_requested_deadline_missed in strtemp) or (errstr_sample_lost in strtemp):
                    error_found = True
                    if dumpallerrors:
                        print("At time: " + output_data[n][0] + " : ", end='')
                        print(strtemp, end='')
                    if errstr_liveliness_changed in strtemp:
                        error_type = np.array([float(output_data[n][0]), float(1), float(0), float(0)])
                    elif errstr_requested_deadline_missed in strtemp:
                        error_type = np.array([float(output_data[n][0]), float(0), float(1), float(0)])
                    elif errstr_sample_lost in strtemp:
                        error_type = np.array([float(output_data[n][0]), float(0), float(0), float(1)])

        if data_set == "train":
            train_outputs.append(error_type)
            train_inputs.append(input_data[i])
        elif data_set == "test":
            test_outputs.append(error_type)
            test_inputs.append(input_data[i])
        # cut off the tail of input data where there is no error
        if n == output_data.shape[0]:
            break

    return np.array(train_inputs), np.array(train_outputs), np.array(test_inputs), np.array(test_outputs)


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

    for p in range(0, data.shape[0]):
        for d in range(0, data[0].shape[0]):
            calc = (data[p][d] - mean[d]) / sd[d]
            std_data[p][d] = float(calc)

    return std_data


def inverse_standard(data, sd, mean):
    real_data = data

    for p in range(0, data.len()):
        for d in range(0, data[0].len()):
            real_data[p][d] = (data[p][d] * sd[d]) + mean[d]

    return real_data


# TODO read more or all training files at once, to get all errors and possible outcomes
config = input_data.read_json("config.json")
input_data, output_data = input_data.get_inputs_and_outputs(config["training_files"], config["rtps_selection"])

training_in, training_out, testing_in, testing_out = outputs_given_inputs(input_data, output_data, int(config["split"]))
# training_in, testing_in = split_data(config["split"], input_data)
# training_out, testing_out = split_data(config["split"], expected_outputs)

input_sd, input_mean = get_sd_mean(training_in)
output_sd, output_mean = get_sd_mean(training_out)
training_in_std = standard_data(training_in, input_sd, input_mean)
training_out_std = standard_data(training_out, output_sd, output_mean)
testing_in_std = standard_data(testing_in, input_sd, input_mean)
testing_out_std = standard_data(testing_out, output_sd, output_mean)


net = RNN.RNN(config, training_in_std, training_out_std)

# training loop
tf.random.set_seed(5)
random_error = tf.random.uniform(shape=[], minval=0, maxval=2)  # random error type
random_timestamp = tf.random.uniform(shape=[], minval=input_data[0][0], maxval=input_data[len(input_data)-1][0])  # random timestamp
if random_error == 0:
    previous_output = np.array(standard_data([random_timestamp, 1, 0, 0], output_sd, output_mean))
elif random_error == 1:
    previous_output = np.array(standard_data([random_timestamp, 0, 1, 0], output_sd, output_mean))
else:
    previous_output = np.array(standard_data([random_timestamp, 0, 0, 1], output_sd, output_mean))

i = 0
for pkt in training_in_std:
    pkt.append(previous_output)
    actual_output = net.produce_output(pkt)
    net.update(pkt, training_out_std[i], actual_output, show=True)
    previous_output = actual_output
    i = i + 1


# testing loop
tf.random.set_seed(5)
random_error = tf.random.uniform(shape=[], minval=0, maxval=2)  # random error type
random_timestamp = tf.random.uniform(shape=[], minval=input_data[0][0], maxval=input_data[len(input_data)-1][0])  # random timestamp
if random_error == 0:
    previous_output = np.array(standard_data([random_timestamp, 1, 0, 0], output_sd, output_mean))
elif random_error == 1:
    previous_output = np.array(standard_data([random_timestamp, 0, 1, 0], output_sd, output_mean))
else:
    previous_output = np.array(standard_data([random_timestamp, 0, 0, 1], output_sd, output_mean))

i = 0
for pkt in testing_in_std:
    pkt.append(previous_output)
    actual_output = net.produce_output(pkt)
    previous_output = actual_output
    i = i + 1
    # TODO better accuracy stat than below, should be closely linked to reward calculation
    actual_inv = inverse_standard(actual_output, output_sd, output_mean)
    expected_inv = testing_out[i]
    print(abs(actual_inv[0]-expected_inv[0]))

# TODO save the weights and the sd/means
