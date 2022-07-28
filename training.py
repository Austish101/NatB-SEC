# handle the training of the DNN

# file data must be fed into the DNN at a variable speed, it will be slower than real time to start with
import input_data


config = input_data.read_json("config.json")
input_list, output_list = input_data.get_inputs_and_output(config["training_files"])


