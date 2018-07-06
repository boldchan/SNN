eta = 0.09 #learning rate
#parameters for rewards
taupre = taupost = 10
Apre = 0.1
Apost = 1.05 * Apre
xb = 0.62#see paper two-trace model P675
yb = 0.66
yc = 0.28
A_plus = 0.86/60
A_minus = 0.25/60
tau_plus = 19
tau_minus = 34
ymax = 180#?

#parameters for eligibility trace
wmax = 5 
c1 = 1 / wmax
c2 = 1 #?


#input layer
vth_i = 1
a_i = 0.2
b_i = 0.025
t_rest_i = 1

#hidden layer
vth_h = 30
tau_ref_h = 4
tau_m_h = 10
t_rest_h = 5
tau_s_h = 5
gmax_h = 30

#output layer
vth_o = 30
tau_ref_o = 4
tau_m_o = 10
t_rest_o = 5
tau_s_o = 5
gmax_o = 30
alpha_o = 100
beta_o = 0.05
gamma_o = 0

#simulation
T = 50
dt = 0.1