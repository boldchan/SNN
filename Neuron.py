import numpy as np
import matplotlib.pyplot as plt
from layers import *
import parameters as p
import pdb

class LIF_Neuron(object):

	#hidden neuron and special neuron

	def __init__(self, dim, T, dt, vth = 30, tau_ref = 4, tau_m = 10, t_rest = 5, gmax = 30, tau_s = 5):
		self.vth = vth
		self.tau_ref = tau_ref # refractory time period
		self.tau_m = tau_m
		self.t_rest = t_rest # initial refrectory time
		self.gmax = gmax
		self.tau_s = tau_s
		self.dim = dim
		self.refract = t_rest * np.ones(self.dim)
		self.spike = np.zeros(self.dim)
		self.PSP = np.zeros(self.dim)
		self.I = np.zeros(self.dim)
		self.v = np.zeros(self.dim)#membrane potential
		self.T = T
		self.dt = dt

	def forward(self, x, w):
		time = np.arange(0, self.T + self.dt, self.dt) # total simulation time
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
						t_rest = t + self.tau_ref
		return self.vm, self.vm > self.vth

	def forward_t(self, t, spike, w):
		'''
		spike: input spike list, if i-th neuron fires spike[i]=1, otherwise 0
		w: weights of connection before this layer [dim_former_layer x dim_current_layer]
		'''
		dPSPdt = np.zeros(self.dim)
		dIdt = np.zeros(self.dim)
		dvdt = np.zeros(self.dim)
		for i in range(self.dim):
			if self.refract[i] > 0:
				self.refract[i] -= 1
			self.spike[i] = 0
			dIdt[i] = -self.I[i]/self.tau_s
			for j, s in enumerate(spike):
				dIdt[i] += w[j][i] * s
			dPSPdt[i] = (self.I[i] - self.PSP[i]) / self.tau_s
			dvdt[i] = (self.PSP[i] - self.v[i]) / self.tau_m
			#update PSP and I
			self.I[i] += dIdt[i]
			self.PSP[i] += dPSPdt[i]
			if self.refract[i] == 0:
				self.v[i] += dvdt[i]
				if self.v[i] > self.vth:
					self.v[i] = 0
					self.spike[i] = 1
					self.refract[i] = self.tau_ref
		return self.v, self.spike


class hidden_neuron(LIF_Neuron):
	def __init__(self, dim):
		LIF_Neuron.__init__(self, dim, p.T, p.dt, vth = p.vth_h, tau_ref = p.tau_ref_h, tau_m = p.tau_m_h, t_rest = p.t_rest_h, gmax = p.gmax_h, tau_s = p.tau_s_h)

class output_neuron(LIF_Neuron):
	def __init__(self, dim):
		LIF_Neuron.__init__(self, dim, p.T, p.dt, vth = p.vth_o, tau_ref = p.tau_ref_o, tau_m = p.tau_m_o, t_rest = p.t_rest_o, gmax = p.gmax_o, tau_s = p.tau_s_o)
		self.alpha = p.alpha_o
		self.beta = p.beta_o
		self.gamma = p.gamma_o
		self.spikeT = np.zeros((dim, int(p.T/p.dt)))


	# def forward(self, x, t):
	# 	LIF_Neuron.forward_t(self, x, t)
	# 	self.spikeT[i, t] = self.spike
	# 	return self.spikeT

	def decode(self, x, w):
		time = np.arange(0, self.T + self.dt, self.dt)
		_, out = self.forward(x, w)
		y_decode = np.zeros((self.dim, len(time)))
		for n in range(self.dim):
			for tf in np.where(out[n] > 0)[0]:
				mask = np.zeros(len(time))
				mask[tf :] = 1
				y_decode[n] += self.alpha * (self.T - time[tf]) / self.T * np.exp(self.beta *(time[tf] - time)) * mask
			y_decode[n] -= self.gamma
		return out, y_decode
	# def decode(self, x):
	# 	time = np.arange(0, self.T + self.dt, self.dt)
	# 	_, out = self.forward(x, self.T, self.dt)
	# 	y_decode = np.zeros((self.dim, len(time)))
	# 	for n in range(self.dim):
	# 		for tf in np.where(out[n] > 0)[0]:
	# 			mask = np.zeros(len(time))
	# 			mask[tf :] = 1
	# 			y_decode[n] += self.alpha * (T - time[tf]) / T * np.exp(self.beta *(time[tf] - time)) * mask
	# 		y_decode[n] -= self.gamma
	# 	return out, y_decode

class special_neuron(LIF_Neuron):
	def __init__(self, dim):
		LIF_Neuron.__init__(self, dim, p.T, p.dt, vth = p.vth_o, tau_ref = p.tau_ref_o, tau_m = p.tau_m_o, t_rest = p.t_rest_o, gmax = p.gmax_o, tau_s = p.tau_s_o)

class input_neuron(object):
	def __init__(self, input_dim = 3):
		self.vth = p.vth_i
		self.dim = input_dim
		self.a = p.a_i
		self.b = p.b_i
		self.spike =np.zeros(input_dim)
		self.v = np.zeros(input_dim)
		self.T = p.T
		self.dt = p.dt
		self.t_rest = p.t_rest_i
		self.refract = self.t_rest * np.ones(self.dim)

	def forward(self, x):
		time = np.arange(0, self.T + self.dt, self.dt) # total simulation time
		self.vm = np.zeros((self.dim, len(time))) # membrane potential
		for n in range(self.dim):
			for i, t in enumerate(time):
				if self.refract[n] > 0 :
					self.refract[n] -=1
				else:
					self.vm[n, i] = self.vm[n, i - 1] + (self.a * x[n] + self.b) * self.dt
				if self.vm[n, i] > self.vth:
					self.refract[n] = self.t_rest
		return self.vm, self.vm > self.vth 

	def forward_t(self, x, t):
		# pdb.set_trace()
		dvdt = np.zeros(self.dim)
		for i in range(self.dim):
			self.spike[i] = 0
			dvdt[i] = self.a*x[i]+self.b
			self.v[i] += dvdt[i] * self.dt
			if self.v[i] > self.vth:
				self.v[i] = 0
				self.spike[i] = 1
		return self.spike


if __name__ == '__main__':
	input_dim = 3
	hidden_dim = 2
	input = np.array([1, 1, 1])

	input_neuron = input_neuron(input_dim)
	hidden_neuron = hidden_neuron(hidden_dim)
	output_neuron = output_neuron(2)

	if True:
	    w1 = 3 * np.random.rand(3,2) - 1
	    w2 = np.random.rand(2,2)
	    #not sure if membrane potential can be negative

	    _, out1T = input_neuron.forward(input)
	    out2T, _ = hidden_neuron.forward(out1T, w1)
	    _, out3T = output_neuron.decode(out2T, w2)
	else:
		w1 = 50 + np.random.rand(3,2)
		w2 = 70 + np.random.rand(2,2)

		out1T = np.zeros((input_dim, int(p.T/p.dt)))
		out2T = np.zeros((hidden_dim, int(p.T/p.dt)))
		out3T = np.zeros((2, int(p.T/p.dt)))

		for i in range(int(p.T/p.dt)):
			out1 = input_neuron.forward_t(input, i * p.dt)
			out1T[:,i] = out1
			out2s, out2 = hidden_neuron.forward_t(i*p.dt, out1, w1)
			out2T[:,i] = out2
			out3s, out3 = output_neuron.forward_t(i*p.dt, out2, w2)
			out3T[:,i] = out3


	plt.subplot(331)
	plt.plot(out1T[0])
	plt.subplot(332)
	plt.plot(out1T[1])
	plt.subplot(333)
	plt.plot(out1T[2])
	plt.subplot(334)
	plt.plot(out2T[0])
	plt.subplot(335)
	plt.plot(out2T[1])
	plt.subplot(337)
	plt.plot(out3T[0])
	plt.subplot(338)
	plt.plot(out3T[1])
	plt.show()
