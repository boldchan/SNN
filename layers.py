import numpy as np
from scipy.ndimage.interpolation import shift

def synapse(x, w, dt, T):
	'''
	Parameters
	-------
	x: output of former layer(input layer or hidden layer), shape N x (T/dt), N is the neuron number of 
    former layer
	w: weights
	b: bias

    output
    -------
    PSP: postsynaptic potential, shape M x (T/dt), M is the neuron number of next layer
	'''
	time = np.arange(0, T + dt, dt)
	gmax = 30
	tao_s = 5
	tf = x * np.array([[i * dt for i in range(x.shape[1])] for j in range(x.shape[0])])
	alpha_t = gmax * time / tao_s * np.exp(1 - time / tao_s)
	PSP = np.zeros((w.shape[1], len(time)))
	for j in range(w.shape[1]):
		for i in range(tf.shape[0]):
			temp = np.zeros(len(time))
			for t in np.where(tf[i] > 0)[0]:
				temp += shift(alpha_t, t, cval = 0)
			PSP[j] += w[i, j] * temp
	return PSP

