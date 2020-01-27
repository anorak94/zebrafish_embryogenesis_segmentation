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
import pywt

def plot_histogram_of_extrema_min_max(r):
    r_max = find_vals_of_r_extrema(r, "max")
    r_min = find_vals_of_r_extrema(r, "min")
    
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist(r_max, bins=15, density=True, alpha=0.6, color='r')
    ax2.hist(r_min, bins=15, density=True, alpha=0.6, color='g')
    
def save_animation(filename, animation_object):# supported format is mp4
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    animation_object.save(filename, writer=writer)


def animate(u, x_, n_c):
    """
    not working in function form yet need to take a second look
    """
    
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2*x_[-1]), ylim=(np.amin(u),1.2*np.amax(u)))
    ax.set_ylabel("w")
    ax.set_xlabel("cell_index_i")
    fig.tight_layout()
    line, = ax.plot([], [], color='steelblue', lw=2)

    def update(num, u, line):
        line.set_data(x_, u[:, num])
        return line,

    ani = animation.FuncAnimation(fig, update, n_c, fargs=[ u, line],
                                  interval=40, blit=True)
    plt.show()

    
    
def plot_fft_of_readout(r, x):
    fft_r = fft(r)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x,fft_r)
    ax.set_title("fft of r vs x")


def plot_peaks_on_signal(r2d, peaks, x_, dx):
    f, ax = plt.subplots()
    ax.plot(x_, r2d)
    ax.scatter(peaks*dx, r2d[peaks], c = "r")
    
def plot_individual_cell(w, u, r, h, t, x, loc):
    f, (ax) = plt.subplots(2, 2, figsize = (18 ,8))
        
    ax[0, 0].plot(t, h[loc, :], c = "k")
    ax[0, 0].set_title("heaviside vs t")
    ax[0, 1].plot(t, u[loc, :], c = "r")
    ax[0, 1].set_title("u vs t")
    ax[1, 0].plot(t, w[loc, :], c = "g")
    ax[1, 0].set_title("w vs t")
    ax[1, 1].plot(x, r, c = "b")
    ax[1, 1].set_title("r vs x")
    
def plot_scaleogram(sig, wavelet="morl", width_max = 201):
    """
    plots scaleogram of using cwt
    cwt : takes a wavelet a compact wave of finite size 
    and computes convolution with signal at varying scales of the wavelet by translating
    it over the signal gives the output as (scale, n_data)
    
    interpretation : dark bands show power of signal darker bands = more power 
    
    advantages : better than using stft as provides information about both freq and time
               : doesnt assume signal is infinite 
               : has significance tests so we can distinguish whether scaleogram is because of signal or just random noise 
               : use instead of quality factor which is for infinite signals 
    """
    coef, freqs=pywt.cwt(sig,np.arange(1,201),wavelet)
    plt.imshow(coef, cmap='PRGn', aspect='auto',
           vmax=abs(coef).max(), vmin=-abs(coef).max())
    plt.show()
    