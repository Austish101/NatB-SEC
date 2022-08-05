#data_store
#holds persistent data of ALL of the input data from a file, if needed.
#https://stackoverflow.com/questions/13034496/using-global-variables-between-files

def init():
    #current index position, only increase on each read (not write).
    global i
    i = 0

    #Matrixes for storing all the input data
    global input_d
    global input_d_label
    input_d = []
    input_d_label = []
    global output_d
    global output_d_label
    output_d = []
    output_d_label = []
    #Assign IP addresses to IDs - can't use IP address string as DNN input
    #as a note - may need to find a way to scramble IDs on every training run
    # so the DNN doesn't learn specific IDs are bad... more trends to each ID...
    global ip_full_list
    global ip_ID
    ip_full_list = []
    ip_ID = []

    global temp_array
    temp_array = []
