import tensorflow as tf


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

    def perceptron(self):

        # tf Graph input
        input_matrix = tf.placeholder(dtype=tf.string, shape=[self.num_input, 1])
        output_expected = tf.placeholder(dtype=tf.string)

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

        correct_pred = tf.equal(tf.argmax(out_layer_addition, 1), tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    reader = NeuralNetwork()