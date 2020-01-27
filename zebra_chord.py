import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, mode
from math import sqrt
import numpy as np
import matplotlib.animation as animation
from scipy import signal
import math
import scipy.optimize as optimize
import scipy.signal as signal
import itertools
from scipy.fftpack import fft, ifft
plt.style.use('seaborn')
import pickle
from scipy import stats
import copy

def return_default_dic():
    param_dic = {"c":15, "n_t":200, "dt": 0.05, "gamma1":10, "n": 5, "m": 5, "tau_2":5, "gamma2":1, "n_c":100, "dx":1.5,
            "v_0": 1.5, "tau_d": 7,"beta":0.05, "u_0": 0.5, "K_u": 0.3, "tol":1e-6, "loc": 4, "tau_w": 2, "tau_c": 5, "delay":5
            ,"sigma_1":0.0026, "sigma_2":0.0026}
    return param_dic

def run_sim_(dic, coupling = True, noise = True, plot=False, plot_r = False, plot_mollifier=False):
    """
    runs simulations and returns results 
    
    param_dic looks like this 
    
    
    """
    u2d = np.ones((dic["n_c"], dic["c"]*dic["n_t"]))
    w2d = np.zeros((dic["n_c"], dic["c"]*dic["n_t"]))
    r2d = np.zeros(dic["n_c"])
    t = np.asarray([i*dic["dt"] for i in range(dic["n_t"]*dic["c"])])
    x_ = np.asarray([i*dic["dx"] for i in range(dic["n_c"])])
    ha2d = return_heaviside(dic["n_c"], dic["n_t"]*dic["c"], dic["dx"], dic["dt"], dic["v_0"])
    
    if noise:
        r_u = norm.rvs(size=u2d.shape , scale=dic["sigma_1"]**2)
        r_w = norm.rvs(size=w2d.shape , scale=dic["sigma_2"]**2)
    else:
        r_u = np.zeros_like(u2d)
        r_w = np.zeros_like(w2d)
    for i in range(dic["c"]*dic["n_t"]-1):
        for cell in range(1, dic["n_c"]-1):
            
            u2d[cell][i+1] = u2d[cell][i] + dic["dt"]*(-u2d[cell][i]*(1/dic["tau_2"])+ (1/dic["tau_2"])*ha2d[cell][i] + r_u[cell][i] )
            if coupling:
                c = dic["beta"]*(w2d[cell-1][i-dic["tau_c"]]+w2d[cell+1][i-dic["tau_c"]])
            else:
                c = 0
            if (i - dic["tau_d"] >= 0):
                w2d[cell][i+1] = w2d[cell][i] + dic["dt"]*(-w2d[cell][i]*(1/dic["tau_w"]) + dic["gamma1"]*(u2d[cell][i]**dic["n"]/(u2d[cell][i]**dic["n"]+dic["K_u"]**dic["n"]))*(1/(1+w2d[cell][i-dic["tau_d"]]**dic["m"])) + c + r_w[cell][i])
            else:
                w2d[cell][i+1] = w2d[cell][i] + dic["dt"]*(-w2d[cell][i]*(1/dic["tau_w"]) + dic["gamma1"]*(u2d[cell][i]**dic["n"]/(u2d[cell][i]**dic["n"]+dic["K_u"]**dic["n"]))*(1)+c + r_w[cell][i])
    
    
    
    for cell in range(1, dic["n_c"]-1):
        mollifier = dirac_delta_approx(u2d[cell], dic["u_0"])
        
        if (plot_mollifier):
            if (cell == dic["loc"]):
                plt.plot(u2d[cell], mollifier)
                plt.title("mollifier vs u")
        for time in range(1, dic["c"]*dic["n_t"]):
            dudt = f_u(time*dic["dt"], u2d[cell][time], cell, dic["tau_2"], dic["dx"], dic["v_0"])
            increment = w2d[cell][time]*mollifier[time]*dudt*dic["dt"]
            r2d[cell]  = r2d[cell]+increment
    
    if (plot):
        f, (ax) = plt.subplots(2, 2, figsize = (18 ,8))
        
        ax[0, 0].plot(t, ha2d[dic["loc"], :], c = "k")
        ax[0, 0].set_title("heaviside vs t")
        ax[0, 1].plot(t, u2d[dic["loc"], :], c = "r")
        ax[0, 1].set_title("u vs t")
        ax[1, 0].plot(t, w2d[dic["loc"], :], c = "g")
        ax[1, 0].set_title("w vs t")
        ax[1, 1].plot(x_, r2d, c = "b")
        ax[1, 1].set_title("r vs x")
    if (plot_r):
        plt.figure(figsize=(6, 4))
        plt.plot(x_, r2d)
        plt.title("r vs x")
    return u2d, w2d, r2d, ha2d, t, x_


def f_u(t, u, j, tau_2, dx , vo ):
    return (1/tau_2)*(-u + int((j*dx - vo*t) > 0))

def dirac_delta_approx(u, u_0, eps = 1e-12):
    d = []
    for ui in u:
        dd  = (1/3.14)*(1/((ui-u_0)**2+eps**2))
        d.append(dd)
    d = np.asarray(d)/np.amax(d)
    return d

def CFL_number(v, dt, dx):
    return v*dt/dx


def return_heaviside(n_c, n_t, dx, dt, v_0):
    ha2d = np.ones((n_c, n_t))
    for i in range(n_t):
        for j in range(n_c):
            ha2d[j][i] = int((j*dx - v_0*i*dt)>0)
    return ha2d


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    """

    if x.ndim != 1:
        raise ValueError ("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError( "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError ("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def get_flatline(sig):
    """
    finds the flatline in the signal by finding all stretches of zeros and then gettting the index 
    of the longest stretch 
    """
    diff = np.diff(copy.deepcopy(sig))
    sub_threshold_indices = diff < 1e-5
    diff[sub_threshold_indices] = 0
    zero_ranges = zero_runs(diff)
    len_zeros = [(z[1]-z[0]) for z in zero_ranges]
    ind = zero_ranges[np.argmax(len_zeros)][0]
    return ind


