import numpy as np
import matplotlib.pyplot as plt
from layers import *
import pdb

class LIF_Neuron(object):

	#hidden neuron and special neuron

	def __init__(self, dim, vth = 30, tau_ref = 4, tau_m = 10, t_rest = 5, gmax = 30, tau_s = 5):
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
						t_rest = t + self.tau_ref
		return self.vm, self.vm > self.vth

	def forward_t(self, x, t, spike):
		'''
		spike: input spike list, if i-th neuron fires spike[i]=1, otherwise 0
		'''
		dPSPdt = np.zeros(self.dim)
		dIdt = np.zeros(self.dim)
		dvdt = np.zeros(self.dim)
		for i in range(self.dim):
			if self.refract[i] > 0:
				self.refract[i] -= 1
			self.spike[i] = 0
			dIdt[i] = -self.I[i]/self.tau_s
			dPSPdt[i] = (self.I[i] - self.PSP[i]) / self.tau_s
			dvdt[i] = (self.PSP[i] - self.v[i]) / self.tau_m
			#update PSP and I
			self.I[i] += dIdt[i]
			self.PSP[i] += dPSPdt[i]
			if self.refract == 0:
				self.v[i] += dvdt[i]
				if self.v[i] > self.vth:
					self.v[i] = 0
					self.spike[i] = 1
					self.refract[i] = self.tau_ref

class hidden_neuron(LIF_Neuron):
	def __init__(self, dim):
		LIF_Neuron.__init__(self, dim, vth = p.vth_h, tau_ref = p.tau_ref_h, tau_m = p.tau_m_h, t_rest = p.t_rest_h, gmax = p.gmax_h, tau_s = p.tau_s_h)

class output_neuron(LIF_Neuron):
	def __init__(self, dim, T, dt, alpha = 5, beta = 0.05, gamma = 0):
		LIF_Neuron.__init__(self, dim, vth = p.vth_o, tau_ref = p.tau_ref_o, tau_m = p.tau_m_o, t_rest = p.t_rest_o, gmax = p.gmax_o, tau_s = p.tau_s_o)
		self.spikeT = np.zeros((dim, int(T/dt)))
		self.T = T
		self.dt = dt
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

	def forward(self, x, t):
		LIF_Neuron.forward_t(self, x, t)
		self.spikeT[i, t] = self.spike

	def decode(self, x):
		time = np.arange(0, self.T + self.dt, self.dt)
		_, out = self.forward(x, self.T, self.dt)
		y_decode = np.zeros((self.dim, len(time)))
		for n in range(self.dim):
			for tf in np.where(out[n] > 0)[0]:
				mask = np.zeros(len(time))
				mask[tf :] = 1
				y_decode[n] += self.alpha * (T - time[tf]) / T * np.exp(self.beta *(time[tf] - time)) * mask
			y_decode[n] -= self.gamma
		return out, y_decode

class special_neuron(LIF_Neuron):
	def __init__(self):
		LIF_Neuron.__init__(self, 1, vth = p.vth_o, tau_ref = p.tau_ref_o, tau_m = p.tau_m_o, t_rest = p.t_rest_o, gmax = p.gmax_o, tau_s = p.tau_s_o)

class input_neuron(object):
	def __init__(self, input_dim, vth = 1):
		self.vth = vth
		self.dim = input_dim
		self.spike =np.zeros(input_dim)
		self.v = np.zeros(input_dim)

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
						t_rest = t + self.tau_ref
		return self.vm, self.vm > self.vth 

	def forward_t(self, x, t):
		dvdt = np.zeros(self.dim)
		for i in range(self.dim):
			self.spike[i] = 0
			dvdt[i] = 0.2*x[i]+0.025
			self.v[i] += dvdt[i]
			if self.v > self.vth:
				self.v = 0
				self.spike[i] = 1

class Output_Neuron(LIF_Neuron):
	def __init__(self, dim, vth = 30, tau_ref = 3, tau_m = 10, t_rest = 5, gmax = 30, tau_s = 5, alpha = 5, beta = 0.05, gamma = 0):
		LIF_Neuron.__init__(self, dim, vth, tau_ref, tau_m, t_rest, gmax, tau_s)
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

	def decode(self, x, T, dt, w):
		time = np.arange(0, T + dt, dt)
		_, out = self.forward(x, T, dt, w)
		y_decode = np.zeros((self.dim, len(time)))
		for n in range(self.dim):
			for tf in np.where(out[n] > 0)[0]:
				mask = np.zeros(len(time))
				mask[tf :] = 1
				y_decode[n] += self.alpha * (T - time[tf]) / T * np.exp(self.beta *(time[tf] - time)) * mask
			y_decode[n] -= self.gamma
		return out, y_decode


if __name__ == '__main__':
	input_dim = 3
	hidden_dim = 2
	input = np.random.rand(input_dim)
	input = np.array([0.2, 0.1, 0.3])
	input_neuron = Input_Neuron(input_dim, tau_ref = 1)
	out, out1 = input_neuron.forward(input, 50, 0.125)
	w = np.random.rand(3,2)
	hidden_neuron = LIF_Neuron(hidden_dim)
	out2, out3 = hidden_neuron.forward(out1, 50, 0.125, w)
	output_neuron = Output_Neuron(2, gamma = 2)
	w2 = np.random.rand(2,2)
	out4, out5 = output_neuron.forward(out3, 50, 0.125, w2)
	_, out6 = output_neuron.decode(out3, 50, 0.125, w2)
	# out = synapse(out1, w, 0.125, 50)
	plt.subplot(731)
	plt.plot(out1[0])
	plt.subplot(732)
	plt.plot(out1[1])
	plt.subplot(733)
	plt.plot(out1[2])
	plt.subplot(734)
	plt.plot(out[0])
	plt.subplot(735)
	plt.plot(out[1])
	plt.subplot(736)
	plt.plot(out[2])
	plt.subplot(737)
	plt.plot(out2[0])
	plt.subplot(738)
	plt.plot(out2[1])
	plt.subplot(7, 3, 10)
	plt.plot(out3[0])
	plt.subplot(7, 3, 11)
	plt.plot(out3[1])
	plt.subplot(7, 3, 13)
	plt.plot(out4[0])
	plt.subplot(7, 3, 14)
	plt.plot(out4[1])
	plt.subplot(7, 3, 16)
	plt.plot(out5[0])
	plt.subplot(7, 3, 17)
	plt.plot(out5[1])
	plt.subplot(7, 3, 19)
	plt.plot(out6[0])
	plt.subplot(7, 3, 20)
	plt.plot(out6[1])
	plt.show()
