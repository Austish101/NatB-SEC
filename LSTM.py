import keras.models
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import numpy as np


class SplitLSTM:
    def __init__(self, config, input_data, output_data):
        self.learning_rate = float(config['learning_rate'])
        self.discount_rate = float(config['discount_rate'])
        self.epsilon = float(config['epsilon'])
        self.loss = config['loss']
        self.output_activation = config['activation']
        self.hidden_activation = "relu"
        self.gamma = 0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.05

        self.input_data, self.output_data = self.shape_data(input_data, output_data)
        # split off the time output and error outputs
        self.output_time, self.output_error, waste = np.split(self.output_data, [1, 4], axis=1)

        self.error_model = self.create_model([self.input_data.shape[1], self.input_data.shape[2]], 3, "softmax", "sigmoid", self.learning_rate, self.loss)
        self.time_model = self.create_model([self.input_data.shape[1], self.input_data.shape[2]], 1, "linear", "relu", self.learning_rate, self.loss)

    def shape_data(self, input_data, output_data):
        new_in = []
        new_out = []
        for i in range(2, input_data.shape[0]):
            new_in.append([input_data[i-2], input_data[i-1], input_data[i]])
            new_out.append(output_data[i])
        # np_in = np.array(new_in)
        return np.array(new_in), np.array(new_out)

    def create_model(self, input_shape, output_shape, output_activation, hidden_activation, learning_rate, loss):
        model = Sequential()
        model.add(LSTM(50, input_shape=(input_shape[0], input_shape[1]), activation=hidden_activation))
        model.add(Dense(output_shape, activation=output_activation))
        model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate))
        return model

    def fit_models(self, epochs=100, batch_size=1, verbose=2):
        self.error_model.fit(self.input_data, self.output_error, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.time_model.fit(self.input_data, self.output_time, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, test_input, test_output):
        test_input_shaped, test_output_shaped = self.shape_data(test_input, test_output)
        error_predict = self.error_model.predict(test_input_shaped)
        time_predict = self.time_model.predict(test_input_shaped)

        test_time, test_error, waste = np.split(test_output, [1, 4], axis=1)

        error_score = self.evaluate_error(error_predict, test_error)
        time_score = self.evaluate_time(time_predict, test_time)

        combined_score = error_score * time_score

        return time_score, error_score, combined_score

    def evaluate_time(self, prediction, expected):
        scores = []
        total = 0
        for i in range(0, prediction.shape[0]):
            score = abs(prediction[i] - expected[i])
            if expected[i] > prediction[i]:
                score = prediction[i] / expected[i]
                return
            elif expected[i] < prediction[i]:
                score = expected[i] / prediction[i]
                return
            elif expected[i] == prediction[i]:
                score = 1
                return

            scores.append(score)
            total = total + score
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

    def save_weights_sd_mean(self, input_sd, input_mean, output_sd, output_mean):
        input_sd_mean = np.array([input_sd, input_mean])
        output_sd_mean = np.array([output_sd, output_mean])
        self.error_model.save("error_model")
        self.time_model.save("time_model")
        np.savetxt('input_sd_mean.txt', input_sd_mean, fmt='%f')
        np.savetxt('output_sd_mean.txt', output_sd_mean, fmt='%f')

    def load_models(self, error_file, time_file):
        self.error_model = keras.models.load_model("error_model")
        self.time_model = keras.models.load_model("time_model")
