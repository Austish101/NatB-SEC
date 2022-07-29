# handle the training of the DNN

# file data must be fed into the DNN at a variable speed, it will be slower than real time to start with
import input_data
import RNN
import tensorflow as tf


# get the expected output given input packets: the next error type and timestamp
def outputs_given_inputs(input_data, output_data):
    output = []
    n = 0
    for i in range(len(input_data)):
        if output_data[n][0] < input_data[i][0]:
            n = n + 1
        # TODO change the append to the error type and timestamp of the error
        output.append(output_data[n][1])
    return output


# split a set of data into two, given a percentage
def split_data(split_percentage, data):
    data_len = len(data)
    split_point = round(data_len * (split_percentage / 100))
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