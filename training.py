# handle the training of the DNN

# file data must be fed into the DNN at a variable speed, it will be slower than real time to start with
import input_data
import RNN
import LSTM
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
                    # if dumpallerrors:
                    #     print("At time: " + output_data[n][0] + " : ", end='')
                    #     print(strtemp, end='')
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


def standard_data(data, sd, mean, kind="all"):
    std_data = data

    if data.shape.__len__() == 1:
        for d in range(0, data.shape[0]):
            calc = (data[d] - mean[d]) / sd[d]
            std_data[d] = float(calc)
        return std_data

    for p in range(0, data.shape[0]):
        if kind == "non-error":
            calc = (data[p][0] - mean[0]) / sd[0]
            std_data[p][0] = float(calc)
        else:
            for d in range(0, data[0].shape[0]):
                calc = (data[p][d] - mean[d]) / sd[d]
                std_data[p][d] = float(calc)

    return std_data


# TODO read more or all training files at once, to get all errors and possible outcomes
config = input_data.read_json("config.json")

try:
    training_in_all = np.loadtxt('training_in.txt', dtype=float)
    training_out_all = np.loadtxt('training_out.txt', dtype=float)
    testing_in_all = np.loadtxt('testing_in.txt', dtype=float)
    testing_out_all = np.loadtxt('testing_out.txt', dtype=float)
except FileNotFoundError:
    training_in_list = []
    training_out_list = []
    testing_in_list = []
    testing_out_list = []
    for file in config["training_files"]:
        file_input_data, file_output_data = input_data.get_inputs_and_outputs(config["training_filepath"], file, config["rtps_selection"])
        training_in, training_out, testing_in, testing_out = outputs_given_inputs(file_input_data, file_output_data, int(config["split"]))
        training_in_list.extend(training_in)
        training_out_list.extend(training_out)
        testing_in_list.extend(testing_in)
        testing_out_list.extend(testing_out)
    training_in_all = np.array(training_in_list)
    training_out_all = np.array(training_out_list)
    testing_in_all = np.array(testing_in_list)
    testing_out_all = np.array(testing_out_list)

    np.savetxt('training_in.txt', training_in_all, fmt='%f')
    np.savetxt('training_out.txt', training_out_all, fmt='%f')
    np.savetxt('testing_in.txt', testing_in_all, fmt='%f')
    np.savetxt('testing_out.txt', testing_out_all, fmt='%f')


input_sd, input_mean = get_sd_mean(training_in_all)
output_sd, output_mean = get_sd_mean(training_out_all)
training_in_std = standard_data(training_in_all, input_sd, input_mean)
training_out_std = standard_data(training_out_all, output_sd, output_mean, "non-error")
testing_in_std = standard_data(testing_in_all, input_sd, input_mean)
testing_out_std = standard_data(testing_out_all, output_sd, output_mean, "non-error")

# using non-std output data?
scores = []
net = LSTM.SplitLSTM(config, training_in_std, training_out_std)
for i in range(0, 100):
    net.fit_models(epochs=10)
    time_score, error_score = net.predict(testing_in_std, testing_out_std)
    scores.append([time_score, error_score])
    print("Time Score:", time_score, "\nError Score:", error_score)
np.savetxt('scores_over_100_by_10_trains.txt', np.array(scores))

# for RNN:
# # net = RNN.RNN(config, training_in_std, training_out_std)
#
# # net.fit(training_in_std, training_out_all)
#
# # training loop
# tf.random.set_seed(5)
# random_error = tf.random.uniform(shape=[], minval=0, maxval=2)  # random error type
# random_timestamp = tf.random.uniform(shape=[], minval=training_in_all[0][0], maxval=training_in_all[training_in_all.shape[0]-1][0])  # random timestamp
# if random_error == 0:
#     previous_output = standard_data(np.array([random_timestamp, 1, 0, 0]), output_sd, output_mean)
# elif random_error == 1:
#     previous_output = standard_data(np.array([random_timestamp, 0, 1, 0]), output_sd, output_mean)
# else:
#     previous_output = standard_data(np.array([random_timestamp, 0, 0, 1]), output_sd, output_mean)
#
# for episodes in range(0, int(config['episodes'])):
#     i = 0
#     for pkt in training_in_std:
#         inputs = np.append(pkt, previous_output)
#         # pkt.append(previous_output)
#         actual_output = net.produce_output(inputs)
#         net.update(inputs, training_out_std[i], actual_output, show=True)
#         previous_output = actual_output
#         i = i + 1
#
#
# # testing loop
# tf.random.set_seed(5)
# random_error = tf.random.uniform(shape=[], minval=0, maxval=2)  # random error type
# random_timestamp = tf.random.uniform(shape=[], minval=file_input_data[0][0], maxval=file_input_data[file_input_data.shape[0]-1][0])  # random timestamp
# previous_output = np.array([0.0, 0.0, 0.0, 0.0])
# # if random_error == 0:
# #     previous_output = np.array(standard_data([random_timestamp, 1, 0, 0], output_sd, output_mean))
# # elif random_error == 1:
# #     previous_output = np.array(standard_data([random_timestamp, 0, 1, 0], output_sd, output_mean))
# # else:
# #     previous_output = np.array(standard_data([random_timestamp, 0, 0, 1], output_sd, output_mean))
#
# i = 0
# for pkt in testing_in_std:
#     pkt.append(previous_output)
#     actual_output = net.produce_output(pkt)
#     previous_output = actual_output
#     i = i + 1
#     # TODO better accuracy stat than below, should be closely linked to reward calculation
#     actual_inv = inverse_standard(actual_output, output_sd, output_mean)
#     expected_inv = testing_out[i]
#     print(abs(actual_inv[0]-expected_inv[0]))

# TODO save the weights and the sd/means
