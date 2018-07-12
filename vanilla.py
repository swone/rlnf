import random
import numpy as np
import start

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def logit(x):
    return np.log(x/(1-x))

def relu(x):
    zero = 0*x
    out = np.maximum(x, zero)
    return out

def elu(x, a):
    if x >= 0:
        return x
    else:
        return a*(np.exp(x)-1)

def generate_data(sample_size):
    data = []
    for i in range(sample_size):
        x = start.generate_random_graph(5, 0.2, 10)
        y = start.edmonds_karp(x)
        x.shape = ([25, 1])
        data.append((x, y))
    return data

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1)+0.2 for y in sizes[1:]]
        self.weights = [np.random.rand(y, x)/20 for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        i = 0
        for b, w in zip(self.biases, self.weights):
            #if i == len(self.biases)-1:
            a = relu(np.dot(w, a)+b)
            #else:
            #    a = sigmoid(np.dot(w, a)+b)
            #i+=1
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            if j%15 == 14:
                eta/=3
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        cost = 0
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            cost+=1/(2*len(mini_batch))*(self.feedforward(x)-y)**2
        print(cost)
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        #i = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            #if i == len(self.biases)-1:
            activation = relu(z)
            #else:
            #    activation = sigmoid(z)
            activations.append(activation)
            #i+=1
        #print(activations[2])
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            #sp = sigmoid_prime(z)
            derv = np.array(z)
            for i in range(len(derv)):
                for j in range(len(derv[0])):
                    if derv[i][j] > 1:
                        derv[i][j] = 1
                    else:
                        derv[i][j] = 0
            delta = np.dot(self.weights[-l+1].transpose(), delta)#* derv# * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())+0.1*self.weights[-l]/(len(self.weights[-1])*len(self.weights[-l][0]))
        return(nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(self.feedforward(x), y)
                for (x, y) in test_data]
        j = 0
        for i in test_results:
            #if j%2 == 0:
            #    print(i[0], i[1])
            j+=1
        return sum(round(x[0][0]) == y for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return(output_activations-y)/10

net = Network([25, 25, 25, 1])

training_data  = generate_data(50)
test_data = generate_data(500)

net.SGD(training_data, 200, 10, 0.01, test_data=training_data)

