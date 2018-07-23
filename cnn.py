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
epochs = 200
batch_size = 50

x = tf.placeholder(tf.float32, [None, 10, 10])
x_shaped = tf.reshape(x, [-1, 10, 10, 1])
y = tf.placeholder(tf.float32, [None, 1])

w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.03), name='w1')
b1 = tf.Variable(tf.truncated_normal([16]), name='b1')
out1 = tf.nn.conv2d(x_shaped, w1, [1, 1, 1, 1], padding='SAME')
out1 += b1
out1 = tf.nn.relu(out1)
out1 = tf.nn.max_pool(out1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
w2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.03), name='w2')
b2 = tf.Variable(tf.truncated_normal([32]), name='b2')
out2 = tf.nn.conv2d(out1, w2, [1, 1, 1, 1], padding='SAME')
out2 += b2
out2 = tf.nn.relu(out2)
out2 = tf.nn.max_pool(out1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
flattened = tf.reshape(out2, [-1, 400])
wd = tf.Variable(tf.truncated_normal([400, 1000], stddev=0.03), name='wd')
bd = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd')
dense = tf.matmul(flattened, wd) + bd
dense = tf.nn.relu(dense)
wout = tf.Variable(tf.truncated_normal([400, 1], stddev=0.03), name='wd')
bout = tf.Variable(tf.truncated_normal([1], stddev=0.01), name='bd')
output = tf.matmul(flattened, wout) + bout
output = tf.nn.relu(output)

cost = tf.losses.mean_squared_error(y, output)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(y, tf.round(output))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

training_data = generate_data(50000)
test_data = generate_data(500)

test_x = [data[0] for data in test_data]
test_y = np.array([data[1] for data in test_data])
test_y.shape = (500, 1)

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = 1000
    for epoch in range(epochs):
        random.shuffle(training_data)
        avg_cost = 0
        for i in range(total_batch):
            batch = training_data[i*batch_size:(i+1)*batch_size]
            batch_x = [data[0] for data in batch]
            batch_y = np.array([data[1] for data in batch])
            batch_y.shape = (batch_size, 1)
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
            avg_cost += c/total_batch
        test_acc = sess.run(accuracy, feed_dict={x: test_x, y:test_y})
        print("Epoch:", (epoch + 1), "cost =", avg_cost,
                "test_accuracy:", test_acc)
    print("\nTraining Complete")
    print(sess.run(accuracy, feed_dict={x: test_x, y:test_y}))