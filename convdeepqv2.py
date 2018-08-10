from collections import deque
import random
import numpy as np
import tensorflow as tf
import start

state_size = [10, 10, 1]
action_size = [100]
learning_rate = 0.00001

total_episodes = 1000
max_steps = 200
max_tau = 50
batch_size = 64

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

gamma = 0.95

pretrain_length = 10000
memory_size = 10000

training = True

episode_render = False

class DQNNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.current = 0

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name="IS_weights")
            
            self.actions_ = tf.placeholder(tf.float32, [None, *action_size], name="actions_")

            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, 
                                        filters=32, 
                                        kernel_size=[10,10],
                                        strides=[1,1],
                                        padding="SAME",
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        name="conv1")
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1, 
                                                                training=True,
                                                                epsilon=1e-10,
                                                                name="batch_norm1")
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, 
                                        filters=64, 
                                        kernel_size=[4,4],
                                        strides=[2,2],
                                        padding="SAME",
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        name="conv2")
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2, 
                                                                training=True,
                                                                epsilon=1e-10,
                                                                name="batch_norm2")
            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")

            self.flat_inputs_ = tf.layers.flatten(self.conv2_out)

            self.full1 = tf.layers.dense(inputs=self.flat_inputs_, units=512,
                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="fc1")
            self.full2 = tf.layers.dense(inputs=self.full1, units=256,
                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="fc2")
            self.output = tf.layers.dense(inputs=self.full1, units=100,
                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            self.absolute_errors = tf.abs(self.target_Q - self.Q)
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
    
tf.reset_default_graph()

DQNetwork = DQNNetwork(state_size, action_size, learning_rate)

TargetNetwork = DQNNetwork(state_size, action_size, learning_rate, name="TargetNetwork")

def update_target_graph():
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    
    return op_holder

class SumTree(object):
    
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity

        self.tree = np.zeros(2*capacity - 1)

        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index-1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]
    
class Memory(object):

    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.6

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.

    def __init__(self, capacity):

        self.tree = SumTree(capacity)

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []
        b_idx, b_ISWeights = np.empty((n, ), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority() / n

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority()

        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i+1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority()

            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e

        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)

        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

memory = Memory(memory_size)

"""class Memory():
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
"""
fgraph = start.generate_random_graph(10, 0.6, 15)
rgraph = np.copy(fgraph)
for i in range(10):
    for j in range(10):
        if fgraph[i][j] > 0:
            if j == 9:
                rgraph[i][j] = 10
            else:
                rgraph[i][j] = 1
        else:
            rgraph[i][j] = -11
possible_actions = []

for i in range(10):
    actions = []
    for j in range(10):
    #action = np.zeros([10, 10])
    #action[i][j] = 1
        if fgraph[i][j] == 0:
            actions.append(0)
        else:
            actions.append(1)
    possible_actions.append(actions)
possible_actions = np.array(possible_actions)
current = 0
path = [0]

for i in range(pretrain_length):
    if i == 0:
        state = np.stack((fgraph), axis=2)
    to = np.random.randint(0, 10)
    action = to
    current = to
    path.append(action)
    possible_actions[action][path[-1]] = 0
    reward = rgraph[path[-1]][action]
    if action == 9:
        minf = 15
        for x in range(len(path)-1):
            minf = min(minf, fgraph[path[x]][path[x+1]])
        if minf == 0:
            minf = -1.1
        reward = 10*minf
    action = np.zeros((10, 10))
    action[path[-2]][to] = 1 
    if reward < 1:
        current = 0
        fgraph = start.generate_random_graph(10, 0.6, 15)
        rgraph = np.copy(fgraph)    
        for i in range(10):
            for j in range(10):
                if fgraph[i][j] > 0:
                    if j == 9:
                        rgraph[i][j] = 10
                    else:
                        rgraph[i][j] = 1
                else:
                    rgraph[i][j] = -11
        possible_actions = []
        for i in range(10):
            actions = []
            for j in range(10):
                if fgraph[i][j] == 0:
                    actions.append(0)
                else:
                    actions.append(1) 
            possible_actions.append(actions)
        possible_actions = np.array(possible_actions)
        next_state = np.stack((np.zeros((10, 10))), axis=2)
        #update flow graph
        memory.store((state, action, reward, next_state, True))
        state = np.stack((fgraph), axis=2)
        path = [0]
    elif reward > 1:
        current = 0
        fgraph = start.generate_random_graph(10, 0.6, 15)
        rgraph = np.copy(fgraph)    
        for i in range(10):
            for j in range(10):
                if fgraph[i][j] > 0:
                    if j == 9:
                        rgraph[i][j] = 10
                    else:
                        rgraph[i][j] = 1
                else:
                    rgraph[i][j] = -11
        possible_actions = []
        for i in range(10):
            actions = []
            for j in range(10):
                if fgraph[i][j] == 0:
                    actions.append(0)
                else:
                    actions.append(1) 
            possible_actions.append(actions)
        possible_actions = np.array(possible_actions)
        next_state = np.stack((np.zeros((10, 10))), axis=2)
        memory.store((state, action, reward, next_state, False))
        state = np.stack((fgraph), axis=2)
        path = [0]
    else:
        next_state = np.stack((fgraph), axis=2)
        memory.store((state, action, reward, next_state, False))
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
        #qs = actions[current] * Qs[0]
        #print(qs)
        choice = np.argmax(Qs[current*10:(current+1)*10])
        action = int(choice)

    return action, explore_probability

saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:

        #saver.restore(sess, "./models/model.ckpt")
        sess.run(tf.global_variables_initializer())

        decay_step = 0

        path = [0]

        fgraph = start.generate_random_graph(10, 0.6, 15)
        rgraph = np.copy(fgraph)
        for i in range(10):
            for j in range(10):
                if fgraph[i][j] > 0:
                    if j == 9:
                        rgraph[i][j] = 10
                    else:
                        rgraph[i][j] = 1
                else:
                    rgraph[i][j] = -11
        possible_actions = []
        for i in range(10):
            actions = []
            for j in range(10):
                if fgraph[i][j] == 0:
                    actions.append(0)
                else:
                    actions.append(1) 
            possible_actions.append(actions)
        possible_actions = np.array(possible_actions)
        current = 0

        state = np.stack((fgraph), axis=2)
        path = [0] 
        tau = 0

        update_target = update_target_graph()

        sess.run(update_target)

        for episode in range(total_episodes):

            step = 0

            episode_rewards = []

            while step < max_steps:
                step+=1
                
                tau+=1

                decay_step+=1

                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate,
                    decay_step, state, path[-1], possible_actions)
                to = action
                current = action
                possible_actions[action][path[-1]] = 0
                reward = rgraph[path[-1]][action]
                path.append(action)
                if to == 9:
                    minf = 15
                    for x in range(len(path)-1):
                        minf = min(minf, fgraph[path[x]][path[x+1]])
                    if minf == 0:
                        minf = -1.1
                    reward = 10*minf
                episode_rewards.append(reward)
                action = np.zeros((10, 10))
                action[to] = 1
                if reward < 1:
                    current = 0
                    fgraph = start.generate_random_graph(10, 0.6, 15)
                    rgraph = np.copy(fgraph)
                    for i in range(10):
                        for j in range(10):
                            if fgraph[i][j] > 0:
                                if j == 9:
                                    rgraph[i][j] = 10
                                else:
                                    rgraph[i][j] = 1
                            else:
                                rgraph[i][j] = -11
                    possible_actions = []
                    for i in range(10):
                        actions = []
                        for j in range(10):
                            if fgraph[i][j] == 0:
                                actions.append(0)
                            else:
                                actions.append(1) 
                        possible_actions.append(actions)
                    possible_actions = np.array(possible_actions)
                    next_state = np.stack((np.zeros((10, 10))), axis=2)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode), 'Total Reward: {}'.format(total_reward),
                        'Training loss: {:.4f}'.format(loss), 'Explore P: {:.4f}'.format(explore_probability))
                    memory.store((state, action, reward, next_state, True))
                    state = np.stack((fgraph), axis=2)
                    path = [0]
                elif reward > 1:
                    current = 0
                    fgraph = start.generate_random_graph(10, 0.6, 15)
                    rgraph = np.copy(fgraph)
                    for i in range(10):
                        for j in range(10):
                            if fgraph[i][j] > 0:
                                if j == 9:
                                    rgraph[i][j] = 10
                                else:
                                    rgraph[i][j] = 1
                            else:
                                rgraph[i][j] = -11
                    possible_actions = []
                    for i in range(10):
                        actions = []
                        for j in range(10):
                            if fgraph[i][j] == 0:
                                actions.append(0)
                            else:
                                actions.append(1) 
                        possible_actions.append(actions)
                    possible_actions = np.array(possible_actions)
                    next_state = np.stack((np.zeros((10, 10))), axis=2)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode), 'Total Reward: {}'.format(total_reward),
                        'Training loss: {:.4f}'.format(loss), 'Explore P: {:.4f}'.format(explore_probability))
                    memory.store((state, action, reward, next_state, True))
                    state = np.stack((fgraph), axis=2)
                    path = [0]
                else:
                    next_state = np.stack((fgraph), axis=2)
                    memory.store((state, action, reward, next_state, False))
                    state = next_state

                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []

                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})
                q_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})

                for i in range(len(batch)):
                    terminal = dones_mb[i]

                    action = np.argmax(q_next_state[i])

                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma*np.max(q_target_next_state[i][action])
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _, absolute_errors = sess.run([DQNetwork.loss, DQNetwork.optimizer, DQNetwork.absolute_errors], feed_dict={DQNetwork.inputs_: states_mb, 
                    DQNetwork.target_Q: targets_mb, DQNetwork.actions_: actions_mb, DQNetwork.ISWeights_: ISWeights_mb})
                memory.batch_update(tree_idx, absolute_errors)
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb, DQNetwork.target_Q: targets_mb,
                    DQNetwork.actions_: actions_mb, DQNetwork.ISWeights_: ISWeights_mb})
                writer.add_summary(summary, episode)
                writer.flush()

                if tau > max_tau:
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            if episode % 1000 == 0:
                save_path = saver.save(sess, "./models/model_2.ckpt")
                print("Model Saved")
                
with tf.Session() as sess:
    total_score = 0
    saver.restore(sess, "./models/model_2.ckpt")
    for i in range(200):
        fgraph = start.generate_random_graph(10, 0.6, 15)
        possible_actions = []
        for i in range(10):
            actions = []
            for j in range(10):
                if fgraph[i][j] == 0:
                    actions.append(0)
                else:
                    actions.append(1) 
            possible_actions.append(actions)
        possible_actions = np.array(possible_actions)
        current = 0
        state = np.stack((fgraph), axis=2)
        path = [0]
        done = False
        score = 0
        while not done:
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
            action = np.argmax(Qs[current*10:(current+1)*10])
            print(action)
            action = int(action)
            current = action
            possible_actions[action][path[-1]] = 0
            reward = rgraph[path[-1]][action]
            path.append(action)
            score+=reward
            if reward < 1:
                done = True
            if reward > 1:
                path = [0]
                current = 0
                fgraph = start.generate_random_graph(10, 0.6, 15)
                possible_actions = []
                for i in range(10):
                    actions = []
                    for j in range(10):
                        if fgraph[i][j] == 0:
                            actions.append(0)
                        else:
                            actions.append(1) 
                    possible_actions.append(actions)
                possible_actions = np.array(possible_actions)
                state = np.stack((fgraph), axis=2)
        total_score+=score
    print(total_score/200.0)