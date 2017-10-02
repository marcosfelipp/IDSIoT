import tensorflow as tf
import csv

class CSVReader:
    def __init__(self):
        self.dataset_list = []
        self.reader()

    def reader(self):
        file = open("../data/kddcup.data10", 'r')
        try:
            tuples_csv = csv.reader(file)
            for linha in tuples_csv:
                self.dataset_list.append(linha)
        finally:
           file.close()

    def normatize_data(self):
        for tuple in self.dataset_list:
            pass

    def print_tuple(self):
        for linha in self.dataset_list:
            print(linha)

class DataAnalysis:
    def __init__(self, total_tuples):
        self.input_tenor = None

        # Network parameters:
        self.n_hidden_1 = 41         # 1st layer number of features
        self.n_input = total_tuples  # Total of datasets
        self.n_class = 24 + 1        # Classes of atacks + 1 normal

        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_class]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b3': tf.Variable(tf.random_normal([self.n_class]))
        }

        self.multilayer_perceptron(self.input_tenor,self.weights, self.biases)

    def perceptron(self, input_tensor, weights, biases):
        layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
        layer_1_addiction = tf.add(layer_1_multiplication, biases['b1'])
        layer_1_activate = tf.nn.relu(layer_1_addiction)

        out_layer_multiplication = tf.matmul(layer_1_activate, weights['out'])
        out_layer_addiction = out_layer_multiplication + biases['out']

        return out_layer_addiction

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
