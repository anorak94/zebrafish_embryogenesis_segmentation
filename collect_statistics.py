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

def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1

def closest_time(u, u_0, t):
    vals = list((u-u_0)**2)
    val = min(vals)
    return int(listRightIndex(vals, val))
    
def find_vals_of_r_extrema(r2d, type_ex=None):
    """
    
    returns values of r extremas 
    """
    if (type_ex == "max"):
        extrema  = signal.argrelextrema(r2d, np.greater)
    else:
        extrema  = signal.argrelextrema(r2d, np.less)
    return r2d[extrema]



def dist_between_peaks(peaks, dx):
    dist = []
    for i in range(1, len(peaks)):
        dist.append(2*dx*(peaks[i]-peaks[i-1]))
    dist = np.asarray(dist)
    stats = np.unique(dist, return_index=False, return_inverse=False, return_counts=True, axis=None)
    return stats, peaks, dist


def extract_somite_statistics_from_readout(r2d, dx):
    peaks, _ = signal.find_peaks(r2d)
    extrema_g,  = signal.argrelextrema(r2d, np.greater)
    extrema_l,  = signal.argrelextrema(r2d, np.less)
    np.append(extrema_l, 0)
    if extrema_g[0]<extrema_l[0]:
        peaks =np.asarray( list(itertools.chain(*zip(list(extrema_g), list(extrema_l))))[:-1])
    else:
        peaks =np.asarray( list(itertools.chain(*zip(list(extrema_l), list(extrema_g))))[:-1])

    return dist_between_peaks(peaks, dx)

def normal_test(data, alpha = 1e-4):
    print ("null hypothesis: data comes from a normal distribution")
    k, p = stats.normaltest(data)
    if (p < alpha): 
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")
    return (k, p, alpha)


def get_oscillations_of_w_single_cell(w2d, t):
    """
    here tmax is max to max and tmin is min to min 
    fix this for tmax to be min to max and tmin to be 
    max to min
    """
    peaks, _ = signal.find_peaks(w2d)
    extrema_g,  = signal.argrelextrema(w2d, np.greater)
    extrema_l,  = signal.argrelextrema(w2d, np.less)
    w_max = w2d[extrema_g]
    w_min = w2d[extrema_l]
    
    np.append(extrema_l, 0)
    if extrema_g[0]<extrema_l[0]:
        peaks =  np.asarray( list(itertools.chain(*zip(list(extrema_g), list(extrema_l))))[:-1])
    else:
        peaks = np.asarray( list(itertools.chain(*zip(list(extrema_l), list(extrema_g))))[:-1])
    t_ex = t[peaks]
    
    
    t_ = np.asarray([x - t_ex[i - 1] for i, x in enumerate(t_ex)][1:])
    t_max, t_min = zigzag(list(t_))
   
    np.append(extrema_l, 0)
    
    return w_max, w_min, t_max, t_min, peaks

def zigzag(seq):
    return seq[::2], seq[1::2]


def return_snr_mean_to_std(signal):
    """
    snr = mean/std
    """
    return np.mean(signal)/np.std(signal)

def signaltonoise_scipy(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def return_CV(signal):
    return 100*np.std(signal)/np.mean(signal)

def calc_power(signal):
    return np.sum(signal**2)/signal.size

def calc_snr_db(signal, sigma):
    """
    calc power of noise = variance of the gaussian white noise
    calc power of signal+noise in time domain
    snr = 10*math.log10(P_s/P_n - 1)
    """
    P_s = calc_power(signal)
    P_n = sigma**2
    return 10*math.log10(P_s/P_n)

def calc_snr_power(signal, sigma):
    """
    calc power of noise = variance of the gaussian white noise
    calc power of signal+noise in time domain
    snr = (P_s/P_n)
    """
    P_s = calc_power(signal)
    P_n = sigma**2
    return (P_s/P_n)