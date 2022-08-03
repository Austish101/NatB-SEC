#test_input_data_main.py

import input_data
import data_store
filename = "D:/Users/Luke/Documents/SEChallenge2022/DataSet/SECH.IncrementalDelayAndLoss_Eth1"

data_store.init()
input_data.get_inputs_and_outputs(filename)
input_list, output_list = input_data.get_next()
print(input_list)
print(output_list)
