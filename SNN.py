import numpy as np
from Neuron import *

class Two_Layer_SNN(object):
    '''
    a two-layer spiking neuron network with a IF-Neuron without leakage as input layer, two LIF-Neuron 
    as hidden and output layer

    The architecture should be input layer - affine - hidden layer - affine - output layer, in which 
    synapse applies first alpha function then affine and its output is the postsynaptic potential (PSP) 
    '''

    def __init__(self, input_dim = 3, hidden_dim, output_dim, T = 50, dt = 0.125, t_rest = 0):
        self.T = T # total time to simulate(ms)
        self.dt = dt # simulation time step(ms)
        self.t_rest = t_rest # initial refrectory time

        self.params = {}

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        W1 = weight_scale * np.random.randn(input_dim, hidden_dim)
        W2 = weight_scale * np.random.randn(hidden_dim, output_dim)
        self.params['W1'] = W1
        self.params['W2'] = W2

        self.input_neuron = Input_Neuron(input_dim, t_rest = self.t_rest)
        self.hidden_neuron = LIF_Neuron(hidden_dim, t_rest = self.t_rest)
        self.output_neuron = Output_Neuron(output_dim, gamma = 2)


    def reward(self, x, y = None):
        '''
        compute loss and weight modification

        Inputs
        --------
        - x: Array of input data of shape (input_dim)
        - y: Array of output data of shape (output_dim)

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - outpu: Array of shape (N, C) output of motor neuron

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to change of those parameters.
        '''

        ####################forward#############################
        h = self.hidden_neuron.forward(x, self.T, self.dt)
        out, y_cap = self.output_neuron.decode(h, self.T, self.dt)

        ####################modification########################
        lr = 1e-2
        time = np.arange(self.dt, self.T + self.dt, self.dt)
        output_layer_reward = np.zeros(self.output_dim)
        hidden_layer_reward = np.zeros(self.hidden_dim)
        if y is None:
            for t in time:
                for i in range(self.output_dim):
                    for j in range(self.hidden_dim):
                            dw = self.STDP()
        else:
            for t in time:
                output_layer_rewards += y - y_cap
                hidden_layer_reward = self.params['W2'] @ output_layer_rewards
                g_w2 = 1 - np.exp(-2 * self.params['W2']/ np.max(self.params['W2']))
                a_pre

        ###tobedone###
    def updateSTDP():
        pass

    def deltaW():
        ##todo##
        pass

    def updateET():
        pass

    def simulate():
        ##todo##
        pass









    def STDP(self, dt):
        A_p = 1
        A_n = 1
        tau_p = 15
        tau_n = 50
        if dt >= 0:
            return A_p * exp(-dt / tau_p)
        else:
            return -A_n * exp(dt / tau_n)

    def R-STDP(self, sj, si, tau_c):
        '''
        Parameters
        -------
        si: presynaptic spike times
        sj: postsynaptic spike times
        tau_c: time constant of eligibility trace

        Output
        -------
        eligibility trace
        '''




