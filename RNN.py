import keras.models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import time


class Split:
    def __init__(self, config, input_data, output_data, input_sd, input_mean, output_sd, output_mean):
        self.learning_rate = float(config['learning_rate'])
        self.discount_rate = float(config['discount_rate'])
        self.epsilon = float(config['epsilon'])
        self.loss = config['loss']
        self.output_activation = config['activation']
        self.hidden_activation = config['hidden_activation']  # "relu"
        self.nodes_in_layer = int(config['nodes_in_layer'])
        self.gamma = float(config['gamma'])
        self.epsilon_min = float(config['epsilon_min'])
        self.epsilon_decay = float(config['epsilon_decay'])
        self.tau = float(config['tau'])
        self.input_sd = input_sd
        self.input_mean = input_mean
        self.output_sd = output_sd
        self.output_mean = output_mean

        self.input_data, self.output_data = self.shape_data(input_data, output_data, int(config['window_size']))
        # split off the time output and error outputs
        self.output_time, self.output_error, waste = np.split(self.output_data, [1, 4], axis=1)

        self.error_model = self.create_model(self.input_data.shape[1], self.output_error.shape[1], "softmax", "sigmoid", self.learning_rate, self.loss)
        self.time_model = self.create_model(self.input_data.shape[1], self.output_time.shape[1], "linear", "relu", self.learning_rate, self.loss)

    # def shape_data(self, input_data, output_data):
    #     new_in = []
    #     new_out = []
    #     for i in range(2, input_data.shape[0]):
    #         new_in.append([input_data[i-2], input_data[i-1], input_data[i]])
    #         new_out.append(output_data[i])
    #     # np_in = np.array(new_in)
    #     return np.array(new_in), np.array(new_out)
    def shape_data(self, input_data, output_data, window_size):
        new_in = []
        new_out = []
        new_index = 0
        for i in range(0, input_data.shape[0], window_size):
            i_input = input_data[i].tolist()
            new_in.append(i_input)
            for n in range(1, window_size):
                try:
                    n_input = input_data[i + n].tolist()
                    new_in[new_index].extend(n_input)
                except IndexError:
                    for e in range(n, window_size):
                        np_zeros = np.zeros(input_data[n-1].shape[0])
                        e_input = np_zeros.tolist()
                        new_in[new_index].extend(e_input)
                    break
            new_index += 1
            new_out.append(output_data[i])
        # np_in = np.array(new_in)
        return np.array(new_in), np.array(new_out)

    def create_model(self, input_shape, output_shape, output_activation, hidden_activation, learning_rate, loss):
        model = Sequential()
        # model.add(LSTM(self.nodes_in_layer, input_shape=(input_shape[0], input_shape[1]), activation=hidden_activation, unroll=True))
        model.add(Dense(self.nodes_in_layer, input_dim=input_shape, activation=hidden_activation))
        model.add(Dense(self.nodes_in_layer, activation=hidden_activation))
        model.add(Dense(self.nodes_in_layer, activation=hidden_activation))
        model.add(Dense(output_shape, activation=output_activation))
        model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate))
        return model

    def fit_models(self, epochs=100, batch_size=1, verbose=2):
        self.error_model.fit(self.input_data, self.output_error, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.time_model.fit(self.input_data, self.output_time, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def unstd_time(self, std_time):
        calc = std_time * self.output_sd[0]
        timestamp = calc + self.output_mean[0]
        return timestamp

    # predict using real data, data must be shaped correctly, specified in config['window']
    def predict_real(self, inputs, error_threshold=0.9, time_threshold=1000):
        error_predict = self.error_model.predict(inputs)
        time_predict = self.time_model.predict(inputs)
        wait_timestamp = 0

        current_timestamp = time.time_ns()
        predicted_timestamp = self.unstd_time(time_predict)
        time_to_error = abs(current_timestamp - predicted_timestamp)

        # if error is above thresholds error type is returned, else -1 is returned, predictions returned for logging
        for n in range(0, error_predict.shape[0]):
            if error_predict[n] > error_threshold and time_to_error < time_threshold:
                n = n + 10
                # TODO API post
                return n, error_predict, time_predict
        return -1, error_predict, time_predict

    def predict_test(self, test_input, test_output, score_or_stats, error_threshold=0.8, time_threshold=500):
        test_time, test_error, waste = np.split(test_output, [1, 4], axis=1)
        wait_timestamp = 0
        num_errors_missed = 0

        # use stats or scoring
        if score_or_stats == "stats":
            predictions = []
            stats = []

            for i in range(0, test_input.shape[0]):
                print(i)
                # maintains correct shape for training
                i_input = [test_input[i:(i+1)]]
                error_predict_shaped = self.error_model.predict(i_input)
                error_predict = error_predict_shaped[0]
                time_predict_shaped = self.time_model.predict(i_input)
                time_predict = time_predict_shaped[0][0]

                current_time = test_input[i][test_input[i].size - 1]
                current_timestamp = self.unstd_time(current_time)
                predicted_timestamp = self.unstd_time(time_predict)
                time_to_error = abs(current_timestamp - predicted_timestamp)

                # if error is above thresholds error type is returned, else -1 is returned, predictions returned for logging
                for n in range(0, error_predict.shape[0]):
                    # ensure error is only sent once
                    if error_predict[n] > error_threshold and time_to_error < time_threshold:
                        print("Prediction at i =", i)
                        n = n + 10
                        wait_timestamp = current_timestamp + time_threshold
                        predicted_error = n
                        predictions.append([predicted_error, error_predict, time_predict])

                        # stats calc - time_difference and error correctness
                        expected_timestamp = self.unstd_time(test_time[i][0])
                        time_difference = abs(expected_timestamp - predicted_timestamp)
                        if time_difference < time_threshold:
                            is_time_correct = True
                        else:
                            is_time_correct = False
                        is_error_correct = False
                        for e in range(0, error_predict.shape[0]):
                            if test_error[i][e] == 1:
                                if predicted_error == e:
                                    is_error_correct = True
                        # if time is after error, error missed
                        if current_timestamp > expected_timestamp:
                            num_errors_missed = num_errors_missed + 1
                        stats.append([is_error_correct, is_time_correct, time_difference, num_errors_missed])
                    # else - if thresholds not met
                    # else:
                    #   predictions.append(np.array([-1, error_predict, time_predict]))

            return np.array(predictions), np.array(stats)

        else:
            error_predict = self.error_model.predict(test_input)
            time_predict = self.time_model.predict(test_input)

            error_score = self.evaluate_error(error_predict, test_error)
            time_score = self.evaluate_time(time_predict, test_time)
            combined_score = error_score * time_score

            return time_score, error_score, combined_score

    def evaluate_time(self, prediction, expected):
        scores = []
        total = 0
        for i in range(0, prediction.shape[0]):
            score = abs(prediction - expected)
            if expected[i] > prediction[i]:
                score = prediction[i] / expected[i]
            elif expected[i] < prediction[i]:
                score = expected[i] / prediction[i]
            elif expected[i] == prediction[i]:
                score = [1]

        time_difference = abs(prediction - expected)
        if expected > prediction:
            score = prediction / expected
        elif expected < prediction:
            score = expected / prediction
        elif expected == prediction:
            score = [1]

            scores.append(score[0])
            total = total + score[0]
        average = float(total / prediction.shape[0])

        return average

    def evaluate_error(self, prediction, expected):
        scores = []
        total = 0
        for i in range(0, prediction.shape[0]):
            score = 0
            for e in range(0, prediction.shape[1]):
                if expected[i][e] == 1:
                    score = prediction[i][e]
                    break
            scores.append(score)
            total = total + score
        average = float(total / prediction.shape[0])

        return average

    # saves the model + sd and mean of the dataset, sd/mean should not be needed if model contains this data
    def save_model_sd_mean(self, tag, input_sd, input_mean, output_sd, output_mean):
        input_sd_mean = np.array([input_sd, input_mean])
        output_sd_mean = np.array([output_sd, output_mean])
        self.error_model.save("error_model%s" % tag)
        self.time_model.save("time_model%s"
                             "" % tag)
        np.savetxt('input_sd_mean%s.txt' % tag, input_sd_mean, fmt='%f')
        np.savetxt('output_sd_mean%s.txt' % tag, output_sd_mean, fmt='%f')

    def load_models(self, error_file, time_file):
        self.error_model = keras.models.load_model(error_file)
        self.time_model = keras.models.load_model(time_file)
