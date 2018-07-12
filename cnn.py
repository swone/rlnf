import random
import start
import numpy as np
import tensorflow as tf

def generate_data(sample_size):
    data = []
    for i in range(sample_size):
        x = start.generate_random_graph(10, 0.8, 10)
        y = start.edmonds_karp(x)
        data.append((x, y))
    return data

learning_rate = 0.0001
epochs = 30
batch_size = 50

x = tf.placeholder(tf.float32, [None, 10, 10])
x_shaped = tf.reshape(x, [-1, 10, 10, 1])
y = tf.placeholder(tf.float32, [None, 1])

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    return out_layer

layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [2, 2], [1, 1], name='layer2')
flattened = tf.reshape(layer2, [-1, 576])
wd = tf.Variable(tf.truncated_normal([576, 1000], stddev=0.03), name='wd')
bd = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd')
dense = tf.matmul(flattened, wd) + bd
dense = tf.nn.relu(dense)
wout = tf.Variable(tf.truncated_normal([576, 1], stddev=0.03), name='wd')
bout = tf.Variable(tf.truncated_normal([1], stddev=0.01), name='bd')
output = tf.matmul(flattened, wout) + bout
output = tf.nn.relu(output)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(y, tf.round(output))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

training_data = generate_data(5000)
test_data = generate_data(500)

test_x = [data[0] for data in test_data]
test_y = np.array([data[1] for data in test_data])
test_y.shape = (500, 1)

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = 100
    for epoch in range(epochs):
        random.shuffle(training_data)
        avg_cost = 0
        for i in range(total_batch):
            batch = training_data[i*batch_size:(i+1)*batch_size]
            batch_x = [data[0] for data in batch]
            batch_y = np.array([data[1] for data in batch])
            batch_y.shape = (batch_size, 1)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x:batch_x, y:batch_y})
            avg_cost += c/total_batch
        test_acc = sess.run(accuracy, feed_dict={x: test_x, y:test_y})
        print("Epoch:", (epoch + 1), "cost =", avg_cost,
                "test_accuracy:", test_acc)
    print("\nTraining Complete")
    print(sess.run(accuracy, feed_dict={x: test_x, y:test_y}))