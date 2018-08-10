from collections import deque
import random
import numpy as np
import tensorflow as tf
import start

state_size = [3, 10, 10]
action_size = [10]
learning_rate = 0.001

total_episodes = 100000
max_steps = 200
batch_size = 32

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

gamma = 0.95

pretrain_length = batch_size
memory_size = 10000

training = True

episode_render = False

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, *action_size], name="actions_")

            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.flat_inputs_ = tf.reshape(self.inputs_, [-1, 3*10*10])

            self.full1 = tf.layers.dense(inputs=self.flat_inputs_, units=300,
                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="fc1")
            self.full2 = tf.layers.dense(inputs=self.full1, units=300,
                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="fc2")
            self.output = tf.layers.dense(inputs=self.full2, units=10,
                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
    
tf.reset_default_graph()

DQNetwork = DQNetwork(state_size, action_size, learning_rate)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                size=batch_size, replace=False)
        return [self.buffer[i] for i in index]

memory = Memory(max_size=memory_size)

fgraph = start.generate_random_graph(10, 0.4, 15)
fgraph.shape = (1, 10, 10)
rgraph = np.copy(fgraph)
for i in range(10):
    for j in range(10):
        if fgraph[0][i][j] > 0:
            if j == 9:
                rgraph[0][i][j] = 10
            else:
                rgraph[0][i][j] = 1
        else:
            rgraph[0][i][j] = -50
possible_actions = []

for i in range(10):
    actions = []
    for j in range(10):
    #action = np.zeros([10, 10])
    #action[i][j] = 1
        if fgraph[0][i][j] == 0:
            actions.append(0)
        else:
            actions.append(1)
    possible_actions.append(actions)
possible_actions = np.array(possible_actions)
possible_actions.shape = (1, 10, 10)
current = np.zeros([1, 10, 10])
current[0][0][0] = 1
path = [0]
for i in range(pretrain_length):
    if i == 0:
        state = np.vstack((fgraph, current, possible_actions))
    to = np.random.randint(0, 10)
    action = to
    if len(path) == 1:
        current[0][0] = 0
        current[0][to] = 1
    else:
        current[path[-2]][path[-1]] = 0
        current[path[-1]][to] = 1
    path.append(action)
    possible_actions[0][action][path[-1]] = 0
    reward = rgraph[0][path[-1]][action]
    print(reward)
    if reward == 10:
        minf = 15
        for x in range(len(path)-1):
            minf = min(minf, fgraph[0][path[x]][path[x+1]])
        reward += 10*minf
    action = np.zeros(10)
    action[to] = 1 
    if reward > -1 or reward < -1:
        nextf = np.zeros((1, 10, 10))
        currentf = np.zeros((1, 10, 10))
        possf = np.zeros((1, 10, 10))
        next_state = np.vstack((nextf, currentf, possf))
        #update flow graph
        memory.add((state, action, reward, next_state, True))
        if reward > 1:
            current[0][path[-2]][path[-1]] = 0
            current[0][0][0] = 1
            fgraph = start.generate_random_graph(10, 0.4, 15)
            fgraph.shape = (1, 10, 10)
            possible_actions = []
            for i in range(10):
                actions = []
                for j in range(10):
                    if fgraph[0][i][j] == 0:
                        actions.append(0)
                    else:
                        actions.append(1) 
                possible_actions.append(actions)
            possible_actions = np.array(possible_actions)
            possible_actions.shape = (1, 10, 10)
            state = np.vstack((fgraph, current, possible_actions))
        else:
            current[0][path[-2]][path[-1]] = 0
            current[0][0][0] = 1
            for x in range(1, len(path)):   
                possible_actions[0][path[x-1]][path[x]] = 1
            state = np.vstack((fgraph, current, possible_actions))
        path = [0]
    else:
        next_state = np.vstack((fgraph, current, possible_actions))
        memory.add((state, action, reward, next_state, False))
        #make sure it can't loop
        state = next_state

writer = tf.summary.FileWriter("/tensorboard/dqn/1")
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, current, actions):
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if(explore_probability > exp_exp_tradeoff):
        action = np.random.randint(0, 10)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

        choice = np.argmax(Qs)
        action = int(choice)

    return action, explore_probability

saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        decay_step = 0

        path = [0]

        fgraph = start.generate_random_graph(10, 0.4, 15)
        fgraph.shape = (1, 10, 10)
        possible_actions = []
        for i in range(10):
            actions = []
            for j in range(10):
                if fgraph[0][i][j] == 0:
                    actions.append(0)
                else:
                    actions.append(1) 
            possible_actions.append(actions)
        possible_actions = np.array(possible_actions)
        possible_actions.shape = (1, 10, 10)
        current = np.zeros((1, 10, 10))
        current[0][0][0] = 1

        state = np.vstack((fgraph, current, possible_actions))
        path = [0]

        for episode in range(total_episodes):

            step = 0
            
            episode_rewards = []

            while step < max_steps:
                step+=1
                decay_step+=1

                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate,
                    decay_step, state, current, possible_actions)
                to = action
                if len(path) > 1:
                    current[0][path[-2]][path[-1]] = 0
                    current[0][path[-1]][action] = 1
                else:
                    current[0][0][0] = 0
                    current[0][0][action] = 1
                possible_actions[0][action][path[-1]] = 0
                reward = rgraph[0][path[-1]][action]
                path.append(action)

                episode_rewards.append(reward)
                action = np.zeros(10)
                action[to] = 1
                if reward > 1 or reward < -1:
                    nextf = np.zeros((1, 10, 10))
                    currentf = np.zeros((1, 10, 10))
                    possf = np.zeros((1, 10, 10))
                    next_state = np.vstack((nextf, currentf, possf))
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode), 'Total Reward: {}'.format(total_reward),
                        'Training loss: {:.4f}'.format(loss), 'Explore P: {:.4f}'.format(explore_probability))
                    memory.add((state, action, reward, next_state, True))
                    if reward > 1:
                        current[0][path[-2]][path[-1]] = 0
                        current[0][0][0] = 1
                        fgraph = start.generate_random_graph(10, 0.4, 15)
                        fgraph.shape = (1, 10, 10)
                        possible_actions = []
                        for i in range(10):
                            actions = []
                            for j in range(10):
                                if fgraph[0][i][j] == 0:
                                    actions.append(0)
                                else:
                                    actions.append(1) 
                            possible_actions.append(actions)
                        possible_actions = np.array(possible_actions)
                        possible_actions.shape = (1, 10, 10)
                        state = np.vstack((fgraph, current, possible_actions))
                    else:
                        current[0][path[-2]][path[-1]] = 0
                        current[0][0][0] = 1
                        for x in range(1, len(path)):   
                            possible_actions[0][path[x-1]][path[x]] = 1
                        state = np.vstack((fgraph, current, possible_actions))
                else:
                    next_state = np.vstack((fgraph, current, possible_actions))
                    memory.add((state, action, reward, next_state, False))
                    state = next_state

                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=2)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                for i in range(len(batch)):
                    terminal = dones_mb[i]

                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma*np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                print(actions_mb.shape)
                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer], feed_dict={DQNetwork.inputs_: states_mb, 
                    DQNetwork.target_Q: targets_mb, DQNetwork.actions_: actions_mb})
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb, DQNetwork.target_Q: targets_mb,
                    DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
                
