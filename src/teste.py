import tensorflow as tf
import numpy as np
from data_manipulation import DataManipulation


class NeuralNetwork:
    def __init__(self):
        # Parameters
        self.learning_rate  = 0.1
        self.num_steps      = 10
        self.batch_size     = 128
        self.display_step   = 100

        # Network Parameters
        self.n_hidden_1     = 10  # 1st layer number of neurons
        self.num_input      = 41
        self.num_classes    = 2  # Normal and anormal

        # Load data
        self.data_reader = DataManipulation()
        self.input_matrix, self.output_matrix = self.data_reader.read_file()

        self.input_len = len(self.output_matrix)

        self.perceptron()

    def perceptron(self):
        print('Starting perceptron')
        # tf Graph input
        input_matrix = tf.placeholder(dtype=tf.float32, shape=[1, self.num_input])
        output_expected = tf.placeholder(dtype=tf.float32, shape=[1, self.num_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, self.num_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

        # Construct model

        layer_1_multiplication = tf.matmul(input_matrix, weights['h1'])
        layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
        layer_1_activation = tf.nn.relu(layer_1_addition)

        out_layer_multiplication = tf.matmul(layer_1_activation, weights['out'])
        out_layer_addition = out_layer_multiplication + biases['out']

        # Compare out value with output expected:
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=out_layer_addition, labels=output_expected))

        # Computing gradients and apply gradients automatic:
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_op)

        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(self.num_steps):
                avg_cost = 0.

                # Loop over all tuples:
                for tuple_position in range(self.input_len):
                    matrix_in = self.transform_tuple_in_matrix(self.input_matrix[tuple_position])
                    print(self.output_matrix[tuple_position])
                    this_x = np.reshape(self.output_matrix[tuple_position], (2, 1))
                    print(this_x)
                    c, _ = sess.run([loss_op, optimizer],
                                    feed_dict={input_matrix: matrix_in,
                                               output_expected: this_x})

                    avg_cost += c / self.input_len

            print("Optimization finished")

        # Test model:
        # correct_pred = tf.equal(tf.argmax(out_layer_addition, 1), tf.argmax(Y, 1))

       #  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def transform_tuple_in_matrix(self, tuple):
        matrix = []
        for var in tuple:
            matrix.append(var)

        return matrix


if __name__ == '__main__':
    reader = NeuralNetwork()