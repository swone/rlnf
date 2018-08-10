import random
import numpy as np
import start

def relu(x):
    zero = 0*x
    out = np.maximum(x, zero)
    return out

def generate_data(sample_size):
    data = []
    for i in range(int(sample_size/4)):
        x = start.generate_random_graph(5, 0.8, 10)
        y = start.edmonds_karp(x)
        x.shape = ([1, 25])
        data.append((x, y))
    for i in range(int(sample_size/4)):
        x = start.generate_random_graph(5, 0.6, 10)
        y = start.edmonds_karp(x)
        x.shape = ([1, 25])
        data.append((x, y))
    for i in range(int(sample_size/4)):
        x = start.generate_random_graph(5, 0.4, 10)
        y = start.edmonds_karp(x)
        x.shape = ([1, 25])
        data.append((x, y))
    for i in range(int(sample_size/4)):
        x = start.generate_random_graph(5, 0.2, 10)
        y = start.edmonds_karp(x)
        x.shape = ([1, 25])
        data.append((x, y))
    return data

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(1, y)+0.1 for y in sizes[1:]]
        self.weights = [np.random.rand(x, y)/20 for x, y in zip(sizes[:-1], sizes[1:])]
        self.nabla_b_p = [np.zeros(b.shape) for b in self.biases]
        self.nabla_w_p = [np.zeros(w.shape) for w in self.weights]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = relu(np.dot(a, w)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            if j==20:
                eta/=10
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            cost = 0
            for x, y in training_data:
                cost+=1/(2*len(training_data))*(self.feedforward(x)-y)**2
            print("cost: ", cost)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j+1, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j+1))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        delta_weights = [(eta/len(mini_batch))*nw+0.75*nwp for w, nw, nwp in zip(self.weights, nabla_w, self.nabla_w_p)]
        delta_biases = [(eta/len(mini_batch))*nb+0.75*nbp for b, nb, nbp in zip(self.biases, nabla_b, self.nabla_b_p)]
        self.weights = [w-dw for w, dw in zip(self.weights, delta_weights)]
        self.biases = [b-db for b, db in zip(self.biases, delta_biases)]
        self.nabla_w_p = delta_weights
        self.nabla_b_p = delta_biases
        

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        #masks = []
        for b, w in zip(self.biases, self.weights):
            """if len(activations) > 1:
                u = np.random.binomial(1, 0.8, activation.shape)/0.8
                activation *= u
                masks.append(u)"""
            z = np.dot(activation, w)+b
            zs.append(z)
            activation = relu(z)
            activations.append(activation)
        delta = 2*(activations[-1]-y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2].transpose(), delta)
        for l in range(2, self.num_layers):
            z = zs[-l]
            derv = np.array(z)
            for i in range(len(derv)):
                for j in range(len(derv[0])):
                    if derv[i][j] > 1:
                        derv[i][j] = 1
                    else:
                        derv[i][j] = 0
            delta = np.dot(delta, self.weights[-l+1].transpose())* derv
        #    if l < self.num_layers-1:
        #        delta *= masks[-l+1]
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(activations[-l-1].transpose(), delta)
        return(nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(self.feedforward(x), y)
                for (x, y) in test_data]
        return sum(round(x[0][0]) == y for (x, y) in test_results)


net = Network([25, 50, 50, 25, 1])

training_data  = generate_data(100000)
test_data = generate_data(1000)

net.SGD(training_data, 500, 50, 0.001, test_data=test_data)

