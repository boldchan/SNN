import numpy as np
import matplotlib.pyplot as plt
from layers import *

class LIF_Neuron(object):

	def __init__(self, dim, vth = 30, tau_ref = 3, tau_m = 10, t_rest = 5, gmax = 30, tau_s = 5):
		self.vth = vth
		self.tau_ref = tau_ref # refractory time period
		self.tau_m = tau_m
		self.t_rest = t_rest # initial refrectory time
		self.v_spike = 5
		self.gmax = gmax
		self.tau_s = tau_s
		self.dim = dim

	def forward(self, x, T, dt, w):
		time = np.arange(0, T + dt, dt) # total simulation time
		self.vm = np.zeros((self.dim, len(time))) # membrane potential
		alpha_t = self.gmax * time / self.tau_s * np.exp(1 - time / self.tau_s)
		for j in range(self.dim):
			PSP = np.zeros(len(time))
			for i in range(x.shape[0]):
				temp = np.zeros(len(time))
				for tf in np.where(x[i] > 0)[0]:
					temp += shift(alpha_t, tf, cval = 0)
				PSP += w[i, j] * temp
			t_rest = self.t_rest
			for i, t in enumerate(time):
				if t > t_rest:
					self.vm[j, i] = self.vm[j, i - 1] + (-self.vm[j, i - 1] + PSP[i - 1])/self.tau_m
					if self.vm[j, i] > self.vth:
						self.vm[j, i] += self.v_spike
						t_rest = t + self.tau_ref
		return self.vm, self.vm > self.vth

class Input_Neuron(object):
	def __init__(self, input_dim, vth = 1, tau_ref = 3, tau_m = 10, t_rest = 0):
		self.vth = vth
		self.tau_ref = tau_ref # refractory time period
		self.tau_m = tau_m
		self.t_rest = t_rest # initial refrectory time
		self.v_spike = 0.2
		self.dim = input_dim

	def forward(self, x, T, dt):
		a = 0.2
		b = 0.025
		time = np.arange(0, T + dt, dt) # total simulation time
		self.vm = np.zeros((self.dim, len(time))) # membrane potential
		for n in range(self.dim):
			t_rest = self.t_rest
			for i, t in enumerate(time):
				if t > t_rest:
					self.vm[n, i] = self.vm[n, i - 1] + (a * x[n] + b) * dt
					if self.vm[n, i] > self.vth:
						self.vm[n, i] += self.v_spike
						t_rest = t + self.tau_ref
		return self.vm, self.vm > self.vth 

class Output_Neuron(object):
	def __init__(self, dim, alpha = 5, beta = 0.05, gamma = 0, t_rest = 0):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.dim = dim
		self.t_rest = t_rest

	def forward(self, x, T, dt):
		time = np.arange(0, T + dt, dt)
		self.vm = np.zeros((self.dim, len(time)))
		for n in range(self.dim):
			for tf in np.where(x[n] > 0)[0]:
				mask = np.zeros(len(time))
				mask[tf :] = 1
				self.vm[n] += self.alpha * (T - time[tf]) / T * np.exp(self.beta *(time[tf] - time)) * mask
			self.vm[n] -= self.gamma
		return self.vm


input_dim = 3
hidden_dim = 2
input = np.random.rand(input_dim)
input = np.array([0.2, 0.1, 0.3])
input_neuron = Input_Neuron(input_dim, tau_ref = 1)
out, out1 = input_neuron.forward(input, 50, 0.125)
w = np.random.rand(3,2)
hidden_neuron = LIF_Neuron(hidden_dim)
out2, out3 = hidden_neuron.forward(out1, 50, 0.125, w)
output_neuron = Output_Neuron(2)
out4 = output_neuron.forward(out3, 50, 0.125)
# out = synapse(out1, w, 0.125, 50)
plt.subplot(531)
plt.plot(out1[0])
plt.subplot(532)
plt.plot(out1[1])
plt.subplot(533)
plt.plot(out1[2])
plt.subplot(534)
plt.plot(out[0])
plt.subplot(535)
plt.plot(out[1])
plt.subplot(536)
plt.plot(out[2])
plt.subplot(537)
plt.plot(out2[0])
plt.subplot(538)
plt.plot(out2[1])
plt.subplot(5, 3, 10)
plt.plot(out3[0])
plt.subplot(5, 3, 11)
plt.plot(out3[1])
plt.subplot(5, 3, 13)
plt.plot(out4[0])
plt.subplot(5, 3, 14)
plt.plot(out4[1])
plt.show()
