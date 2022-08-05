#test_input_data_main.py

import input_data
import data_store
filename = "D:/Users/Luke/Documents/SEChallenge2022/DataSet/SECH.IncrementalDelayAndLoss_Eth1"

data_store.init()
print("###### Calling get_inputs_and_outputs #####################")
input_data.get_inputs_and_outputs(filename)
print("###### Calling get_next ###################################")
print("## input_list, output_list = input_data.get_next()")
input_list, output_list = input_data.get_next()
print("###### Printing First Value from get_next #################")
print("## print(input_list)")
print(input_list)
print("## print(output_list)")
print(output_list)
print("###### Printing First Value direct from data_store ########")
print("## print(data_store.input_d[0])")
print(data_store.input_d[0])
print("## print(data_store.output_d[0])")
print(data_store.output_d[0])
