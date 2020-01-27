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
from zebra_chord import smooth
import scipy
import pywt
import kPyWavelet as wavelet

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

def get_interpeak_distance(sig, distance):
    peaks, _ = scipy.signal.find_peaks(sig)
    return np.diff(distance[peaks])


def get_qf_list(sig_list, cell_loc, type_method):
    if cell_loc == None:
        r_list = sig_list
    else:
        r_list = (np.asarray(sig_list)[:, cell_loc, :])
    qf_W = []
    for w in r_list:
        sample_signal = w
        if(type_method=="linear_interp"):
            qf = compute_qf_linear_interp(sample_signal)
        else:
            qf = curve_intersection_qf(sample_signal)
        qf_W.append(qf[0])
    return qf_W

def compute_qf_linear_interp(sig):
    freq, psd = scipy.signal.welch(sig)
    psd = 10*np.log10(psd)
    sig_max = np.amax(psd)
    y_line = sig_max - 3
    ind_g = np.where(psd > y_line)
    ind_l = np.where(psd < y_line)
    val_g = psd[ind_g]
    val_l = psd[ind_l]

    y_max = np.amax(val_g)
    y_2_max = np.amax(val_l)
    f_2_max = freq[np.where(psd == y_2_max)]
    f_max = freq[np.argmax(psd)]

    f_x = (f_max - f_2_max)*(y_line - y_max)/(y_max - y_2_max) + f_max
    y_x = (f_x - f_max)*(-y_2_max + y_max)/(f_max - f_2_max) + y_max
    qf = f_max/(2*(f_max - f_x))[0]
    return qf, y_x, y_max, y_2_max

def curve_intersection_qf(signal):
    freq, psd = scipy.signal.welch(signal)
    psd = 10*np.log10(psd)
    sig_max = np.amax(psd)
    y_line = sig_max - 3
    idx = np.argwhere(np.diff(np.sign(psd - y_line))).flatten()
    f_max = freq[np.argmax(psd)]
    freq_id = freq[idx[1]]
    return (f_max/2*(f_max-freq_id))

def plot_psd(sig, size_tup, type_plot, plot_cutoff=False):
    freq, psd = scipy.signal.welch(sig)
    fig, ax = plt.subplots(figsize=size_tup)
    psd = 10*np.log10(psd)
    ymax = np.amax(psd)
    if (plot_cutoff):
        ax.plot(freq,[ymax-3 for i in range(len(freq))])
    
    ax.set_ylabel("PSD")
    ax.set_xlabel("freq")
    if (type_plot=="line"):
        ax.plot(freq, psd)
    elif (type_plot=="scatter"):
        ax.scatter(freq, psd)
    else:
        ax.plot(freq, psd)
        ax.scatter(freq, psd)
    return fig, ax


def compute_qf_plotted(w_max):
    freq, psd = scipy.signal.welch(w_max)
    psd = 10*np.log10(psd)
    sig_max = np.amax(psd)
    fig, ax = plt.subplots(figsize=(24, 12))
    
    ymax_below_cutoff = np.amax(psd[np.where(psd<sig_max-3)])

    fm = freq[np.argmax(psd)]
    f2m = freq[np.where(psd == ymax_below_cutoff)]
    ym = sig_max
    y2m = ymax_below_cutoff
    yi = sig_max-3
    fi = (f2m-fm)*(yi-ym)/(y2m-ym)+fm
    yii = (fi-fm)*(y2m-ym)/(f2m-fm) + ym
    
    ax.plot(freq, psd)
    ax.scatter(freq, psd)

    ax.plot(freq, [sig_max for i in range(len(freq))], c = "g")
    ax.plot(freq, [sig_max-3 for i in range(len(freq))], c = "r")
    ax.plot(freq, [ymax_below_cutoff for i in range(len(freq))], c = "k")
    ax.plot(freq, [yii for i in range(len(freq))], c = "b")
    return (fm/2*(fm-fi))


def compute_significance_wavelet(sig, noise):
    """
    computes the significance of the wavelet analysis 
    the wavelet power is distributed as a 2dof chi2 distribution 
    takes as input a normalized signal computes the cwt using morlet wavelet 
    transform 
    
    follows the reference - 
    
    Significance tests for the wavelet power and the wavelet power
    spectrum
    Z. Ge et. al.
    """
    
    sig = sig/np.amax(sig)
    coef, freqs=pywt.cwt(sig,np.arange(1,201),'morl')
    power = np.square(coef)
    power = np.sum(power)/(power.shape[0]*power.shape[1])
    p_t1 = power*2/(noise**2)
    pval = 1-scipy.stats.chi2.cdf(p_t1, df=2)
    return pval


def compute_signif_tc(data): 
    slevel = 0.95  # Significance level
    std = data.std()                      # Standard deviation
    std2 = std ** 2                      # Variance
    var = (data - data.mean()) / std       # Calculating anomaly and normalizing

    dj = 0.5                            # Four sub-octaves per octaves
    s0 = -1 #2 * dt                      # Starting scale, here 6 months
    J = -1 # 7 / dj                      # Seven powers of two with dj sub-octaves
    dt = 0.25
    N = data.size

    alpha = np.corrcoef(var[0:-1], var[1:])[0,1]; 

    mother = wavelet.Morlet(6.)          # Morlet mother wavelet with wavenumber=6

    # The following routines perform the wavelet transform and siginificance
    # analysis for the chosen data set.

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data, dt, dj, s0, J,
                                                          mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother)
    power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
    fft_power = std2 * abs(fft) ** 2     # FFT power spectrum
    period = 1. / freqs

    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                            significance_level=slevel, wavelet=mother)
    return min(signif)

def compute_mu_std(data):
    mu, std = norm.fit(data)
    return mu, std

def normalTest(data):
    k2, p = stats.normaltest(dist_realizations)
    alpha = 1e-3
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print(i, "The null hypothesis can be rejected => x does not come from a normal distribution")
    else:
        print(i, "The null hypothesis cannot be rejected => x might come from a normal distribution")

def get_pval_list(w_l, cell_loc):
    pval_w = []
    for i, w in enumerate(w_l):
        if (cell_loc != None):
            sig_w = compute_signif_tc(w[cell_loc])
        else:
            sig_w = compute_signif_tc(w)
        pval_w.append(sig_w)
    return pval_w

def get_peak_distribution(w_l, cell_loc):
    avg_d = []
    for wi in w_l:
        if cell_loc == None:
            sig = wi
        else:
            sig = wi[cell_loc]
        ind = get_flatline(sig)
        dist = get_interpeak_distance( sig[:ind],t[:ind])
        avg_dist = np.average(dist[:-1])
        avg_d.append(avg_dist)
    return avg_d


def compute_mu_std(data):
    mu, std = norm.fit(data)
    return mu, std

def normalTest(data):
    k2, p = stats.normaltest(data)
    alpha = 1e-3
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print(p, "The null hypothesis can be rejected => x does not come from a normal distribution")
    else:
        print(p, "The null hypothesis cannot be rejected => x might come from a normal distribution")