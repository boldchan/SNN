import numpy as np
from Neuron import *
import parameters as p

class Two_Layer_SNN(object):
    '''
    a two-layer spiking neuron network with a IF-Neuron without leakage as input layer, two LIF-Neuron 
    as hidden and output layer

    The architecture should be input layer - affine - hidden layer - affine - output layer, in which 
    synapse applies first alpha function then affine and its output is the postsynaptic potential (PSP) 
    '''

    def __init__(self, input_dim = 3, hidden_dim = 3, output_dim = 2, T = 50, dt = 0.125, t_rest = 0):
        self.T = T # total time to simulate(ms)
        self.dt = dt # simulation time step(ms)
        self.t_rest = t_rest # initial refrectory time

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_neuron = input_neuron(input_dim)
        self.hidden_neuron = hidden_neuron(hidden_dim)
        self.output_neuron = output_neuron(output_dim)
        self.special_neuron = special_neuron(1)

        W1 = np.array([1 + np.random.randn() for x in range(int(self.input_dim*self.hidden_dim*0.8))]
            +[-(1+np.random.randn()) for x in range(self.input_dim*self.hidden_dim-int(self.input_dim*self.hidden_dim*0.8))])
        W1 = W1.reshape((self.input_dim, self.hidden_dim))
        np.random.shuffle(W1)
        W2 = np.array([1 + np.random.randn() for x in range(0,int(self.output_dim*self.hidden_dim*0.8))]
            +[-(1 + np.random.randn()) for x in range(self.output_dim*self.hidden_dim-int(self.hidden_dim*self.output_dim*0.8))])
        W2 = W2.reshape((self.hidden_dim, self.output_dim))
        np.random.shuffle(W2)
        W3 = np.random.randn(input_dim, 1)
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3

        self.g1 = np.zeros_like(W1)
        self.g2 = np.zeros_like(W2)
        self.g3 = np.zeros_like(W3)

        self.STDP1 = np.zeros_like(W1)
        self.STDP2 = np.zeros_like(W2)
        self.STDP3 = np.zeros_like(W3)

        self.reward1 = np.zeros_like(W1)
        self.reward2 = np.zeros_like(W2)
        self.reward3 = np.zeros_like(W3)

        self.eta = p.eta #learning rate, ignore decay for now


    # def reward(self, x, y = None):
    #     '''
    #     compute loss and weight modification

    #     Inputs
    #     --------
    #     - x: Array of input data of shape (input_dim)
    #     - y: Array of output data of shape (output_dim)

    #     Returns:
    #     If y is None, then run a test-time forward pass of the model and return:
    #     - outpu: Array of shape (N, C) output of motor neuron

    #     If y is not None, then run a training-time forward and backward pass and
    #     return a tuple of:
    #     - loss: Scalar value giving the loss
    #     - grads: Dictionary with the same keys as self.params, mapping parameter
    #       names to change of those parameters.
    #     '''

    #     ####################forward#############################
    #     v1, h1 = self.input_dim.forward(x, self.T, self.dt) #output of input layer
    #     v2, h2 = self.hidden_neuron.forward(h1, self.T, self.dt) #output of hidden layer
    #     _, y_LR = self.output_neuron.decode(h2, self.T, self.dt) #output of output y_L and y_R
    #     _, y_GA = self.special_neuron.decode(h1, self.T, self.dt)#special neuron

    #     ####################calculate reward########################
    #     # for i in range(0, self.hidden_dim):
    #     #     self.

    #     lr = 1e-2
    #     time = np.arange(self.dt, self.T + self.dt, self.dt)
    #     output_layer_reward = np.zeros(self.output_dim)
    #     hidden_layer_reward = np.zeros(self.hidden_dim)
    #     if y is None:
    #         for t in time:
    #             for i in range(self.output_dim):
    #                 for j in range(self.hidden_dim):
    #                         dw = self.STDP()
    #     else:
    #         for t in time:
    #             output_layer_rewards += y - y_cap
    #             hidden_layer_reward = self.params['W2'] @ output_layer_rewards
    #             g_w2 = 1 - np.exp(-2 * self.params['W2']/ np.max(self.params['W2']))
    #             a_pre

        ###tobedone###
    def updateSTDP1(self, h1, h2):
        '''
        h1:former layer output(spike)
        h2:latter layer output(spike)
        based on paper two-trace model for spike-Timing-Dependent synaptic plasticity
        '''
        apre1 = np.zeros((self.input_dim, self.hidden_dim))
        apost1 = np.zeros((self.input_dim, self.hidden_dim))
        STDP1 = np.zeros((self.input_dim, self.hidden_dim))
        for t in range(int(self.T / self.dt) - 1):
            for i in range(self.input_dim):
                for j in range(self.hidden_dim):
                    apre1[i][j] -= apre1[i][j]/p.taupre * self.dt
                    apost1[i][j] -=apost1[i][j]/p.taupost * self.dt
                    if(h1[i][t]):
                        apre1[i][j] += (p.xb>apre1[i][j])*(1-apre1[i][j]/p.xb)
                        STDP1[i][j] -= p.A_minus/p.yc*apre1[i][j]*apost1[i][j]
                    if(h2[j][t]):
                        apost1[i][j] += (apre1[i][j] + p.yc)* (p.yb>apost1[i][j]) * (1 - apost1[i][j]/p.yb)
                        STDP1[i][j] += p.A_plus*apre1[i][j]*(apost1[i][j] - p.yc)*(apost1[i][j]>p.yc)

        self.STDP1 = STDP1

    def updateSTDP2(self, h1, h2):
        '''
        h1:former layer output(spike)
        h2:latter layer output(spike)
        based on paper two-trace model for spike-Timing-Dependent synaptic plasticity
        '''
        apre1 = np.zeros((self.hidden_dim, self.output_dim))
        apost1 = np.zeros((self.hidden_dim, self.output_dim))
        STDP2 = np.zeros((self.hidden_dim, self.output_dim))
        for t in range(int(self.T / self.dt) - 1):
            for i in range(self.hidden_dim):
                for j in range(self.output_dim):
                    apre1[i][j] -= apre1[i][j]/p.taupre * self.dt
                    apost1[i][j] -=apost1[i][j]/p.taupost * self.dt
                    if(h1[i][t]):
                        apre1[i][j] += (p.xb>apre1[i][j])*(1-apre1[i][j]/p.xb)
                        STDP2[i][j] -= p.A_minus/p.yc*apre1[i][j]*apost1[i][j]
                    if(h2[j][t]):
                        apost1[i][j] += (apre1[i][j] + p.yc)* (p.yb>apost1[i][j]) * (1 - apost1[i][j]/p.yb)
                        STDP2[i][j] += p.A_plus*apre1[i][j]*(apost1[i][j] - p.yc)*(apost1[i][j]>p.yc)

        self.STDP2 = STDP2


###################### Update Weights #########################
    def updateET(self): #Eligibility trace
        c1 = p.c1
        c2 = p.c2
        wmax = p.wmax
        self.g1 = 1 - c1*self.W1*np.exp((-1*c2/wmax)*abs(self.W1))
        self.g2 = 1 - c1*self.W2*np.exp((-1*c2/wmax)*abs(self.W2))
        self.g3 = 1 - c1*self.W3*np.exp((-1*c2/wmax)*abs(self.W3))

    # Function called at the end of each iteration
    # Caution - reward, STDP, g1 must have same size as W
    def calculate_deltaW(self):
        self.updateET() # Update the eligliblity trace first
        deltaW1 = self.eta*self.reward1*self.STDP1*self.g1
        self.W1 = np.clip(self.W1+deltaW1, -p.wmax, p.wmax)

        deltaW2 = self.eta*self.reward2*self.STDP2*self.g2
        self.W2 = np.clip(self.W2+deltaW2, -p.wmax, p.wmax)

        deltaW3 = self.eta*self.reward3*self.STDP3*self.g3
        self.W3 = np.clip(self.W3+deltaW3, -p.wmax, p.wmax)



####################### Update rewards #########################

    def update_rewards(self, out, expected):
        '''
        out is the output of snn, expected is the expected value
        '''
        rewardL = (expected[0] - out[0])/p.ymax#?
        rewardR = (expected[1] - out[1])/p.ymax#?
        self.reward2[:,0] = rewardL
        self.reward2[:,1] = rewardR

        for i in range(self.hidden_dim): #this can be made compact
            reward = (abs(self.W2[i][0])*rewardL + abs(self.W2[i][1])*rewardR) / (abs(self.W2[i][0]) + abs(self.W2[i][1]))
            self.reward1[:,i] = reward

    def train(self):
        ##todo##
        data = load_data()
        num_data = len(data['input'])
        for i in range(num_data):
            d = data['input'][i]
            alpha = data['output'][i]
            _, out1 = self.input_neuron.forward(d)
            _, out2 = self.hidden_neuron.forward(out1, self.W1)
            _, out3 = self.output_neuron.decode(out2, self.W2)
            out = [out3[0][-1], out3[1][-1]]
            self.update_rewards(out, alpha)
            self.updateSTDP1(out1, out2)
            self.updateSTDP2(out2, out3)
            self.calculate_deltaW()
            print(self.W2)
            # print(out - alpha)

    def test(self, input):
        '''
        input: g_ypos, g_xneg, g_yneg
        '''
        _, out1 = self.input_neuron.forward(input)
        _, out2 = self.hidden_neuron.forward(out1, self.W1)
        _, out3 = self.output_neuron.decode(out2, self.W2)
        print([out3[0][-1], out3[1][-1]])


def load_data():
    input_data = []
    output_data = []
    with open('target_data.txt', 'r') as f:
        for line in f:
            data_str = line[:-1].split(',')
            data_float = [float(x) for x in data_str]
            data_float /= np.sqrt(data_float[0]*data_float[0] + data_float[1] * data_float[1])
            data_rel = [0, 0, 0]
            if data_float[1] > 0:
                data_rel[0] = data_float[1]
            else:
                data_rel[2] = -data_float[1]
            if (data_float[0] < 0):
                    data_rel[1] = -data_float[0]
            input_data.append(data_rel)
            output_data.append(data_float[2:])
    return {'input':input_data, 'output':output_data}

if __name__ == '__main__':
    snn = Two_Layer_SNN()
    snn.train()
    snn.test([0.7356, 0, 0])



