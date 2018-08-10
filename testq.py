import tensorflow as tf
import numpy as np
import start

with tf.Session() as sess:
    total_score = 0
    saver.restore(sess, "./models/model_2.ckpt")
    for i in range(200):
        fgraph = start.generate_random_graph(10, 0.4, 15)
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
        current = np.zeros((10, 10))
        current[0][0] = 1
        state = np.stack((rgraph, fgraph, current, possible_actions), axis=2)
        path = [0]
        done = False
        score = 0
        while not done:
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
            action = np.argmax(Qs[0])
            action = int(action)
            if len(path) > 1:
                current[path[-2]][path[-1]] = 0
                current[path[-1]][action] = 1
            else:
                current[0][0] = 0
                current[0][action] = 1
            possible_actions[action][path[-1]] = 0
            reward = rgraph[path[-1]][action]
            path.append(action)
            score+=reward
            if reward < 1:
                done = True
            if reward > 1:
                path = [0]
                current = np.zeros((10, 10))
                current[0][0] = 1
                fgraph = start.generate_random_graph(10, 0.4, 15)
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
                state = np.stack((rgraph, fgraph, current, possible_actions), axis=2)
        total_score+=score
    print(total_score/200.0)