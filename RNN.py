from collections import deque

import numpy as np
import random
import csv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from keras.optimizers import Adam


class RNN:
    def __init__(self, config, input_data, output_data):
        self.memory = deque(maxlen=2000)

        self.learning_rate = float(config['learning_rate'])
        self.discount_rate = float(config['discount_rate'])
        self.epsilon = float(config['epsilon'])
        self.loss = config['loss']
        self.activation = config['activation']
        self.gamma = 0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.05

        self.input_data = input_data
        self.output_data = output_data

        # DeepMind suggests using 2 models, one for prediction and one for actual values
        # todo probably not relevant here, to be looked into
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=(self.input_data[0].shape[0]+4), activation="relu"))  # +2 as the last output will be added
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.output_data[0].shape[0], activation=self.activation))
        model.compile(loss=self.loss, optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, inputs, outputs, reward):
        self.memory.append([inputs, outputs, reward])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            inputs, outputs, reward = sample
            target = np.argmax(self.target_model.predict(inputs))
            # future = max(self.target_model.predict(next_inputs)[0])
            target[0][outputs] = reward * self.gamma
            self.model.fit(inputs, target, epoch=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def produce_output(self, inputs):
        # epsilon greedy with decay
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        rand = random.uniform(0, 1)
        if rand <= self.epsilon:
            return [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        return np.argmax(self.model.predict(inputs))

    def fit(self, inputs, outputs, epochs=1000):
        self.model.fit(inputs, outputs, epochs=epochs)

    def update(self, inputs, expected_outputs, actual_outputs, show=False):
        # todo calculate reward based on expected vs actual output, this is a BAD example
        reward = abs(expected_outputs[0] - actual_outputs[0])
        if show:
            print(reward)

        self.remember(inputs, actual_outputs, reward)
        self.replay()
        self.target_train()

    # todo funcations for saving and loading the neural net weights, makes the model portable!
    # TODO also make sure standard deviation and mean of dataset are saved, sd/mean
    def save_data(self, filename):
        with open(filename, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    def read_data(self, filename):
        print("Reading in saved q-table...")
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')