import tensorflow as tf


class NeuralNetwork:
    def __init__(self, input_training, output_training, input_test, output_test):
        # Parameters
        self.learning_rate  = 0.01
        self.num_steps      = 10
        self.batch_size     = 128
        self.display_step   = 100

        # Network Parameters
        self.n_hidden_1     = 10  # 1st layer number of neurons
        self.num_input      = 41
        self.num_classes    = 2  # Normal and anormal

        # Load data
        self.input_matrix   = input_training
        self.output_matrix  = output_training
        self.input_test     = input_test
        self.output_test    = output_test

        self.input_train_len= len(self.output_matrix)
        self.input_test_len = len(self.input_test)

        self.perceptron()

    def perceptron(self):
        '''
        Percetron is Alg ML with 1 label
        :return: None
        '''

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
                for tuple_position in range(self.input_train_len):
                    # Correct shape's problem -- convert vector in matrix
                    input_m = []
                    input_m.append(self.input_matrix[tuple_position])

                    c, _ = sess.run([loss_op, optimizer],
                                    feed_dict={input_matrix: input_m,
                                               output_expected: self.output_matrix[tuple_position]})

                    avg_cost += c / self.input_train_len
                print("Accurace in epoch ", epoch, " : ",avg_cost)

            print("Optimization finished")

            # Test model
            correct_pred = tf.equal(tf.argmax(out_layer_addition, 1), tf.argmax(output_expected, 1))

            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            result = 0
            for tuple in range(self.input_test_len):
                input_m = []
                input_m.append(self.input_test[tuple])
                result+= accuracy.eval({input_matrix: input_m, output_expected: self.output_test[tuple]})

            print("RESULT:",result/self.input_test_len)