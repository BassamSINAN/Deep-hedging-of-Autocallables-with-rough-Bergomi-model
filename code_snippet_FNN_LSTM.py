#Code snippet for FNN and LSTM : example with Heston

import numpy as np
import math
import numpy.random as npr
from scipy.stats import norm
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as kb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, 
                          steps, Npaths, return_vol=False):
    np.random.seed(12345)
    dt = T/steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0,0]), 
                                           cov = np.array([[1,rho],
                                                          [rho,1]]), 
                                           size=Npaths) * np.sqrt(dt) 
        
        S_t = S_t*(np.exp( (r- 0.5*v_t)*dt+ np.sqrt(v_t) *WT[:,0] ) ) 
        v_t = np.abs(v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t)*WT[:,1])
        prices[:, t] = S_t
        sigs[:, t] = v_t
        prices_transpose = np.transpose(prices)
        sigs_transpose = np.transpose(sigs)
        
    if return_vol:
        return prices, sigs

    return prices

def loss_util(lamb,k,option_type):
    def loss(y_true,y_pred):
        gain = kb.sum(y_pred * y_true,axis=-1)
        ST = S_0 + kb.sum(y_true, axis=-1)
        var = kb.abs(y_pred[:,1:T]-y_pred[:,0:T-1])   
        tc = (kb.cumsum(y_true, axis=1)+S_0)*k
        tc_0 = k * S_0
        cost = kb.abs(y_pred[:,0])*tc_0+kb.abs(y_pred[:,T-1])*tc[:,T-1] + kb.sum(tc[:,0:T-1]*var, axis=1)
        
        c = S_0*C
        indicator_AC = tf.math.maximum(tf.math.sign(y_pred[:,int(T/2):int(T/2) +1] - B_AC),0.) # 1 if Autocalls at 6M, 0 otherwise
        pmax = tf.math.reduce_max(kb.cumsum(y_true, axis=1)+S_0,axis=-1) #running maximum
        pmin = tf.math.reduce_min(kb.cumsum(y_true, axis=1)+S_0,axis=-1) #running minimum
        indicator_AC_cont = tf.math.maximum(tf.math.sign(pmax - B_AC),0.) #1 if running max>B_AC, 0 otherwise
        indicator_DIP = tf.math.maximum(tf.math.sign(B_DIP-pmin),0.) # 1 if running_minimum<B_DIP, 0 otherwise
        
        option_list = {
        'call' : kb.maximum(S_0 + kb.sum(y_true,axis=-1) - K,0.),
        'put' : kb.maximum(-S_0 - kb.sum(y_true,axis=-1) + K,0.),
        'cuo' : kb.maximum(S_0 + kb.sum(y_true,axis=-1) - K,0.)*tf.math.maximum(tf.math.sign(B_AC-pmax),0.),
        'autocall' : (S_0 + c)*(indicator_AC) \
                   +(S_0 + c)*(1-indicator_AC) -kb.maximum(K - S_0 - kb.sum(y_true,axis=-1),0.)\
                   *(indicator_DIP)*(1-indicator_AC), #Pays back performance at maturity if doesn't autocalls at 6M
        'autocall_continuous' : (S_0 + c)*(indicator_AC_cont) \
                   +(S_0 + c)*(1-indicator_AC_cont) -kb.maximum(K - S_0 - kb.sum(y_true,axis=-1),0.)\
                   *(indicator_DIP)*(1-indicator_AC_cont), #Autocall obs date is continuous
        }
        
        liability = option_list[option_type]
        pnl = gain - liability - cost
        return tf.math.expm1(-lamb*pnl)/lamb
    return loss

#Heston params choice 
kappa =3
theta = 0.04
v_0 =  sigma**2 # v_t is variance, hence sigma 0 squared
xi = 0.6
r = 0
S_0 = 1
T = 100
rho = -0.8

#Simulating price process
N = 100000
steps = 101
Ti = np.tile(np.linspace(0, 1, T+1)[0:T], (N, 1))
S_HEST = generate_heston_paths(S_0, 1, r, kappa, theta,v_0, rho, xi, steps, N)
dS_HEST = np.diff(S_HEST, 1, 1)
X_HEST = np.stack([Ti, S_HEST[:,0:T]], axis=-1)

#Risk aversion and transaction costs
K=1
risk_aversion = (1,5,10,15) #lambda
cost_level = (0,0.0005,0.005,0.05) #transactions costs

#FNN
F1_HEST, F5_HEST, F10_HEST, F15_HEST = [], [], [], []
F_HEST = [F1_HEST,F5_HEST,F10_HEST,F15_HEST] #for each lambda
for i in range(len(risk_aversion)):
    for j in range(len(risk_aversion)):
        F_HEST[i].append(keras.Sequential([
        keras.layers.InputLayer(input_shape=(T, 2)),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation=output_activation_layer)]))
        
#LSTM
F1_HEST, F5_HEST, F10_HEST, F15_HEST = [], [], [], []
F_HEST = [F1_HEST, F5_HEST, F10_HEST, F15_HEST]
LSTM1_HEST, LSTM5_HEST, LSTM10_HEST, LSTM15_HEST = [], [], [], []
LSTM_HEST = [LSTM1_HEST,LSTM5_HEST,LSTM10_HEST,LSTM15_HEST]
for i in range(len(risk_aversion)): #range lambda
    for j in range(len(risk_aversion)): #range cost level
    prices = keras.layers.Input(shape=(T,2))
        LSTM_HEST[i].append(keras.layers.LSTM(units, return_sequences=True))
        layer_dense = keras.layers.Dense(1, activation="softplus")
        y = LSTM_HEST[i][j](prices)
        hedges = layer_dense(y)
        F_HEST[i].append(keras.models.Model(inputs=prices, outputs=hedges))

#Example with lambda=1 and transactions costs = 0
#Training the network
i, j = 0, 0
F_HEST[i][j].compile(optimizer='adam', loss=loss_util(risk_aversion[i],cost_level[j],option_type), metrics=[])
F_HEST[i][j].fit(X_HEST, dS_HEST, batch_size=batch_size, epochs=epochs)

#Indiff Pricing
loss = F_HEST[i][j].evaluate(X_HEST,dS_HEST)
result = (1/risk_aversion[i] * np.log(risk_aversion[i]*np.float_(loss)+1))
k_cost = cost_level[j]*100
print(r'Risk Aversion=%1.f' % risk_aversion[i], r'Cost level=%1.2f' % k_cost + "%", r'Indiff pricing=%1.22f' % result)


