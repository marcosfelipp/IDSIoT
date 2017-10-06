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
num_classes = 4

# tf Graph input
X = tf.placeholder(dtype=tf.string, shape=[num_input, 1])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model

layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])

out_layer = tf.matmul(layer_1, weights['out']) + biases['out']


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=X))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(out_layer, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    with open(TRAINING) as file:
        for line in file:
            vector = line.strip().split(',')
            sess.run()



