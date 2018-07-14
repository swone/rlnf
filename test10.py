import tensorflow as tf
import random
import numpy as np
import start

def generate_data(sample_size):
    data = []
    for i in range(int(sample_size/4)):
        x = start.generate_random_graph(10, 0.8, 10)
        y = start.edmonds_karp(x)
        x.shape = ([100])
        data.append((x, y))
    for i in range(int(sample_size/4)):
        x = start.generate_random_graph(10, 0.6, 10)
        y = start.edmonds_karp(x)
        x.shape = ([100])
        data.append((x, y))
    for i in range(int(sample_size/4)):
        x = start.generate_random_graph(10, 0.4, 10)
        y = start.edmonds_karp(x)
        x.shape = ([100])
        data.append((x, y))
    for i in range(int(sample_size/4)):
        x = start.generate_random_graph(10, 0.2, 10)
        y = start.edmonds_karp(x)
        x.shape = ([100])
        data.append((x, y))
    return data

training_data = generate_data(50000)
test_data = generate_data(5000)

x = tf.placeholder(tf.float32, [None, 100])
y = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([100, 100], stddev = 0.03))
b1 = tf.Variable(tf.random_normal([100]))
w2 = tf.Variable(tf.random_normal([100, 50], stddev = 0.03))
b2 = tf.Variable(tf.random_normal([50]))
w3 = tf.Variable(tf.random_normal([50, 50], stddev = 0.03))
b3 = tf.Variable(tf.random_normal([50]))
w4 = tf.Variable(tf.random_normal([50, 1], stddev = 0.03))
b4 = tf.Variable(tf.random_normal([1]))
hidden1 = tf.matmul(x, w1)+b1
hidden1 = tf.nn.relu(hidden1)
hidden2 = tf.matmul(hidden1, w2)+b2
hidden2 = tf.nn.relu(hidden2)
hidden3 = tf.matmul(hidden2, w3)+b3
hidden3 = tf.nn.relu(hidden3)
#hidden3 = tf.nn.dropout(hidden3, keep_prob)
y_ = tf.matmul(hidden3, w4)+b4
y_ = tf.nn.relu(y_)
cost = tf.losses.mean_squared_error(y, y_)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
init_op = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.round(y_), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 50

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = 1000
    test_x = [data[0] for data in test_data]
    test_y = np.array([data[1] for data in test_data])
    test_y.shape = ([5000, 1])
    for epoch in range(200):
        avg_cost = 0
        random.shuffle(training_data)
        for i in range(total_batch):
            batch_x = [data[0] for data in training_data[batch_size*i:batch_size*(i+1)]]
            batch_y = np.array([data[1] for data in training_data[batch_size*i:batch_size*(i+1)]])
            batch_y.shape = ([50, 1])
            _, c = sess.run([optimizer, cost], 
                        feed_dict={x: batch_x, y: batch_y, keep_prob: 0.9})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", avg_cost)
    print(sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.0}))