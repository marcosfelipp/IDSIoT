import tensorflow as tf
import os

TRAINING = os.path.join(os.path.dirname(__file__), "../data/kddcup_data.csv")

vector = []

# Parameters
learning_rate = 0.1
num_steps = 10
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons

num_input = 41
num_classes = 2 # Normal and anormal

# tf Graph input
input_matrix = tf.placeholder(dtype=tf.string, shape=[num_input, 1])
output_expected = tf.placeholder(dtype=tf.string)

# Store layers weight & bias
weights = {
    'h1' : tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model

layer_1_multiplication   = tf.matmul(input_matrix, weights['h1'])
layer_1_addition         = tf.add(layer_1_multiplication, biases['b1'])
layer_1_activation       = tf.nn.relu(layer_1_addition)

out_layer_multiplication = tf.matmul(layer_1_activation, weights['out'])
out_layer_addition       = out_layer_multiplication + biases['out']


# Compare out value with output expected:
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer_addition, labels=output_expected))

# Computing gradients and apply gradients automatic:
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(out_layer_addition, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    with open(TRAINING) as file:
        for line in file:
            vector = line.strip().split(',')
            sess.run()



