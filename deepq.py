from collections import deque
import random
import numpy as np
import tensorflow as tf
import start

state_size = [10, 10]
action_size = [10, 10]
learning_rate = 0.001

total_episodes = 1000
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

            self.full1 = tf.layers.dense(inputs=self.inputs_, units=200,
                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="fc1")
            self.full2 = tf.layers.dense(inputs=self.full1, units=100,
                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="fc2")
            self.output = tf.layers.dense(inputs=self.full2, units=100,
                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="fc2")
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.output))
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
rgraph = np.copy(fgraph)
for i in range(10):
    for j in range(10):
        if fgraph[i][j] > 0:
            if j == 9:
                rgraph[i][j] = 1
            else:
                rgraph[i][j] = -1
        else:
            rgraph[i][j] = -50
possible_actions = []

for i in range(10):
    for j in range(10):
        #action = np.zeros([10, 10])
        #action[i][j] = 1
        possible_actions.append((i, j))

for i in range(pretrain_length):
    if i == 0:
        state = fgraph
    path = [0]
    action = random.choice(possible_actions)
    path.append(action[1])
    reward = rgraph[action[0]][action[1]]
    if reward == 1:
        minf = 15
        for x in range(len(path)-1):
            minf = min(minf, fgraph[path[x]][path[x+1]])
        reward += 10*minf
    if reward > -1 or reward < -1:
        next_state = np.zeros(state.shape)
        #update flow graph
        memory.add((state, action, reward, next_state, True))
        state = start.generate_random_graph
    else:
        next_state = state
        memory.add((state, action, reward, next_state, False))
        #make sure it can't loop
        state = next_state

writer = tf.summary.FileWriter("/tensorboard/dqn/1")
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if(explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability

saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        decay_step = 0

        for episode in range(total_episodes):

            step = 0
            
            episode_rewards = []

            state = start.generate_random_graph(10, 0.4, 15)

            while step < max_steps:
                step+=1
                decay_step+=1

                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate,
                    decay_step, state, possible_actions)

                reward = rgraph[action[0]][action[1]]

                episode_rewards.append(reward)

                if reward > 1 or reward < -1:
                    next_state = np.zeros(state.shape)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode), 'Total Reward: {}'.format(total_reward),
                        'Training loss: {:.4f}'.format(loss), 'Explore P: {:.4f}'.format(explore_probability))
                    memory.add((state, action, reward, next_state, True))
                else:
                    next_state = state
                    memory.add((state, action, reward, next_state, False))
                    state = next_state

                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=2)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=2)
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
                
