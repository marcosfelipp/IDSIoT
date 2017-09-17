import tensorflow as tf
import csv


class CSVReader:
    def __init__(self):
        self.list = []
        self.reader()

    def reader(self):
        file = open("../data/kddcup.data10", 'r')
        try:
            tuples_csv = csv.reader(file)
            for linha in tuples_csv:
                self.list.append(linha)
        finally:
            file.close()

    def print_tuple(self):
        for linha in self.list:
            print(linha[3])


class DataAnalysis:
    def __init__(self):
        # Network parameters:
        self.input_tenor = None
        self.n_hidden_1 = 10
        self.n_hidden_2 = 5
        self.n_input = 30
        self.n_class = 4
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_class]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([self.n_class]))
        }

        self.multilayer_perceptron(self.input_tenor,self.weights, self.biases)

    def multilayer_perceptron(self, input_tensor, weights, biases):
        layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
        layer_1_addiction = tf.add(layer_1_multiplication, biases['b1'])
        layer_1_activate = tf.nn.relu(layer_1_addiction)

        layer_2_multiplication = tf.matmul(layer_1_activate, weights['h2'])
        layer_2_addiction = tf.add(layer_2_multiplication, biases['b2'])
        layer_2_activate = tf.nn.relu(layer_2_addiction)

        out_layer_multiplication = tf.matmul(layer_2_activate, weights['out'])
        out_layer_addiction = out_layer_multiplication + biases['out']

        return out_layer_addiction
