# handle the training of the DNN

# file data must be fed into the DNN at a variable speed, it will be slower than real time to start with
import input_data
import RNN
import tensorflow as tf
import numpy as np
#required for clustering (if it works)
import tensorflow_model_optimization as tfmot
import tempfile
import zipfile
import os
import time

#This is required if CUDA drivers are installed, LSTM performs badly on CUDA over CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    liveliness_count = 0
    deadline_count = 0
    sample_count = 0

    # normalise timestamps a bit
    real_ts = time.time()
    start_ts = float(input_data[0][0])
    ts_diff = real_ts - start_ts

    for i in range(len(input_data)):
        # timestamp setup, if difference in recorded and real time is over 1mil seconds, reset
        in_data_ts = float(input_data[i][0]) + ts_diff
        if in_data_ts > (real_ts + 1000000) or in_data_ts < (real_ts - 1000000):
            ts_diff = real_ts - float(input_data[i][0])
            in_data_ts = float(input_data[i][0]) + ts_diff

        if n == output_data.shape[0]:
            n = 0
            continue
        out_data_ts = float(output_data[n][0]) + ts_diff
        if (float(output_data[n][0]) < input_data[i][0]) or (n == 0):
            error_found = False
            error_type = None
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
                        # to even the number of errors in training
                        if data_set == "train":
                            if (liveliness_count <= deadline_count) or (liveliness_count <= sample_count):
                                error_type = np.array([out_data_ts, float(1), float(0), float(0)])
                                liveliness_count += 1
                        else:
                            error_type = np.array([out_data_ts, float(1), float(0), float(0)])
                    elif errstr_requested_deadline_missed in strtemp:
                        # to even the number of errors in training
                        if data_set == "train":
                            if (deadline_count <= liveliness_count) or (deadline_count <= sample_count):
                                error_type = np.array([out_data_ts, float(0), float(1), float(0)])
                                deadline_count += 1
                        else:
                            error_type = np.array([out_data_ts, float(0), float(1), float(0)])
                    elif errstr_sample_lost in strtemp:
                        # to even the number of errors in training
                        if data_set == "train":
                            if (sample_count <= liveliness_count) or (sample_count <= deadline_count):
                                error_type = np.array([out_data_ts, float(0), float(0), float(1)])
                                sample_count += 1
                        else:
                            error_type = np.array([out_data_ts, float(0), float(0), float(1)])
        if error_type is not None:
            in_data = input_data[i]
            in_data[0] = in_data_ts
            if data_set == "train":
                train_outputs.append(error_type)
                train_inputs.append(in_data)
            elif data_set == "test":
                test_outputs.append(error_type)
                test_inputs.append(in_data)
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
    sd = np.std(data, axis=0, dtype=float)
    mean = np.mean(data, axis=0, dtype=float)
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
    print("Training data loaded from saved files")
except:  # FileNotFoundError: (doesn't work on all OS, now just any exception...)
    print("Loading training data from pcap files and saving for faster reading next time, this may take some time")
    training_in_list = []
    training_out_list = []
    testing_in_list = []
    testing_out_list = []
    for file in config["training_files"]:
        file_input_data, file_output_data = input_data.get_inputs_and_outputs(config["training_filepath"], file, config["rtps_selection"])
        training_in, training_out, testing_in, testing_out = outputs_given_inputs(file_input_data, file_output_data, int(config['split']))
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
net = RNN.Split(config, training_in_std, training_out_std, input_sd, input_mean, output_sd, output_mean)
for i in range(0, int(config["episodes"])):
    net.fit_models(epochs=int(config["epochs"]))
    input_shaped, output_shaped = net.shape_data(training_in_std, training_out_std, int(config['window_size']))
    error_threshold = float(config['error_threshold'])
    time_threshold = float(config['time_threshold'])
    # time_score, error_score, combined_score = net.predict_test(input_shaped, output_shaped, "score", error_threshold, time_threshold)
    # scores.append([time_score, error_score])
    # print("Time Score:", time_score, "\nError Score:", error_score, "\nCombined:", combined_score)
    num_correct = 0
    num_time = 0
    num_error = 0
    tot_time_dif = 0

    predictions, stats = net.predict_test(input_shaped, output_shaped, "stats", error_threshold, time_threshold)
    error_correct, time_correct, time_difference, num_errors_missed = np.split(stats, [1, 2, 3], axis=1)
    num_predictions = error_correct.shape[0]
    for n in range(0, num_predictions):
        if error_correct[n] and time_correct[n]:
            num_correct += 1
            num_error += 1
            num_time += 1
        elif error_correct[n]:
            num_error += 1
        elif time_correct[n]:
            num_time += 1
        tot_time_dif = tot_time_dif + time_difference[n]

    avg_time_dif = tot_time_dif / num_predictions
    per_correct = (num_correct / num_predictions) * 100
    per_time = (num_time / num_predictions) * 100
    per_error = (num_error / num_predictions) * 100

    run_data = np.array([per_correct, per_time, per_error, avg_time_dif, np.max(num_errors_missed)])
    np.savetxt('stats_over_%s_by_%s_trains.txt' % (i, int(config["epochs"])), run_data)

# save the weights and the sd/means
net.save_model_sd_mean("%sx%s" % (int(config["episodes"]), int(config["epochs"])), input_sd, input_mean, output_sd, output_mean)

# im going to try SOME CLUSTERING BABYYYYYYYYYYYYYYYYYYYYYYYYYYYYY - Jack Roberto 2k22
# will need to run command
# "pip install -q tensorflow-model-optimization"
# in order to get optimization API and make this work

cluster_weights=tfmot.clustering.keras.cluster_weights
centroidInitialization = tfmot.clustering.keras.centroidInitialization

clustering_params={
    'number_of_clusters': 10,  #too many clusters increases accuracy - slows down system (think nodes)
    'cluster_centroids_init': centroidInitialization.LINEAR
    }
clustered_model = cluster_weights(net, **clustering_params)  # TODO i THINK "model" needs to be replaced with model name
opt = tf.keras.optimizer.Adam(learning_rate=1e-5)
clustered_model.compile(
    loss=tf.keras.losses.sparseCategoricalCrossentropy(from_logits=True),
    optimizer = opt,
    metrics = ['accuracy'])
clustered_model.summary

final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

##advice still dictates that transfering "final model" to tensorflow lite allows the model to run better on restricted hardware
##example (i think) below
clustered_keras_file=tempfile.mkstemp('.h5')
print('saving clustered model to:', clustered_keras_file)
tf.keras.models.save_model(final_model, clustered_keras_file, include_optimizer = False)
#create compressible model for TFLite.
clustered_tflite_file='/tmp/clustered_model.tflite'
converter=tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_clustered_model=converter.convert()
with open(clustered_tflite_file, 'wb') as f:
    f.write(tflite_clustered_model)
    print('saved clustered tflite model to:', clustered_tflite_file)


