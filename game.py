import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import start

ex_graph = [[0, 3, 0, 3, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0], [3, 0, 0, 1, 2, 0, 0], 
        [0, 0, 0, 0, 2, 6, 0], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 9], [0, 0, 0, 0, 0, 0, 0]]
source = 0
sink = 6

ex_graph_1, source_1, sink_1 = start.generate_random_graph(10, 0.6, 15)

print(start.ford_fulkerson(ex_graph_1, source_1, sink_1))
print(start.edmonds_karp(ex_graph_1, source_1, sink_1))

class flow_graph():
    def __init__(self, graph, source, sink):
        self.graph = graph
        self.net_flow = [0]*len(graph[0])
        self.source = source
        self.sink = sink
        self.current = source
        self.score = np.infty

    def getState(self):
        return self.graph

    def getMove(self, action):
        if self.current == self.sink:
            return self.score
        move = int(action*len(self.graph[0]))
        if self.graph[self.current][move] == 0:
            return -100
        else:
            self.score = min(self.score, self.graph[self.current][move])
            self.current = move
            return -1    

class agent():
    def __init__(self, lr):
        self.state_in = tf.placeholder(shape=[10, 10], dtype = tf.int32)
        output = slim.fully_connected(state_in, 10, biases_intitalizer=None,
                activaation_fn = tf.nn.sigmoid, weights_initializer=tf.one_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
        self.update = optimizer.minimize(self.loss)

tf.reset_default_graph()

fGraph = flow_graph()
jBond = agent(lr = 0.001)
weights = tf.trainable_variables()[0]

total_episodes = 10000
total_reward = np.zeros([1])
e = 0.1

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        s = fGraph.getState()

        if np.random.rand(1) < e:
            action = np.random.rand(1)
        else:
            action = sess.run(jBond.chosen_action, 
                feed_dict={jBond.state_in:[s]})
        reward = fGraph.getMove(action)

        feed_dict = {jBond.reward_holder:[reward], 
            jBond.action_holder:[action], jBond.state_in:[s]}
        _,ww = sess.run([jBond.update, weights], feed_dict=feed_dict)

        total_rewards[s, action] += reward
        if i % 500 == 0:
            


"""def game(numV):
    start.generate_random_graph(numV, 0.6, 15)
    net_flow = [0]*numV
    finish = False
    while not finish:
        print("Move")
"""