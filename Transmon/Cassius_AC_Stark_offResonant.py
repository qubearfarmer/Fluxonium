import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.optimize import curve_fit
from scipy.signal import peak_widths, find_peaks
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber

h=6.626e-34
chi = 5e6
kappa = 3.5e6
cavity_freq = 7.5167e9
def IQ_rotate(signal):
    #receives a signal and rotate it to real axis.
    demod_real = np.real(signal)
    demod_imag = np.imag(signal)
    guess_line = [(np.max(demod_imag)-np.min(demod_imag))/(np.max(demod_real)-np.min(demod_real)), np.min(demod_imag)]
    opt,cov = curve_fit(line,xdata=demod_real,ydata=demod_imag, p0 = guess_line)
    theta = np.arctan(opt[0])
    demod_real_rotate = demod_real*np.cos(theta) + demod_imag*np.sin(theta)
    demod_imag_rotate = -demod_real * np.sin(theta) + demod_imag * np.cos(theta)
    return demod_real_rotate + 1j*demod_imag_rotate

def line(x,slope,offset):
    return slope*x - offset

def lorentzian(xdata, amp, center, width,offset):
    x = (center - xdata)/(width/2.0)
    return amp*(1+x**2)**-1.0 + offset

def gaussian_func(x,amp,center,std, offset):
    return amp*np.exp(-(x-center)**2/std**2)-offset

def acStarkShift(power,attenuation, detune):
    photon_flux = power/(attenuation*h*cavity_freq)
    n = photon_flux*kappa/2*((kappa/2)**2 + ((detune+chi)/2)**2)**-1
    return n*chi

def acStarkDephasing (power,attenuation,gamma_residue, detune):
    photon_flux = power / (attenuation * h * cavity_freq)
    n = photon_flux * kappa / 2 * ((kappa / 2) ** 2 + ((detune+chi) / 2) ** 2) ** -1
    return kappa/2 * np.real(np.sqrt((1+1j*chi/kappa)**2 + 4j*chi*n/kappa)-1) + gamma_residue

##################################################################################
#One tone superposition, sweep cavity power
path = 'G:\Projects\Fluxonium\Data\Cassius I\\2019\\09\Data_0917\Two_tone_offResonant_Stark_Out.hdf5'
f = Labber.LogFile(path)
d = f.getEntry(0)
for (channel, value) in d.items():
    print(channel, ":", value)
print ("Number of entries: ", f.getNumberOfEntries())
signal = f.getData('Signal Demodulation - Value')[:-8,:]
qubit_freq = f.getData('Qubit RF - Frequency')[0]
pump_power = f.getData('R&S IQ 2 - Power')[:-8,0]

to_plot = np.zeros(signal.shape)
freq = np.zeros(len(pump_power))
freq_err = np.zeros(len(pump_power))
width = np.zeros(len(pump_power))
width_err = np.zeros(len(pump_power))

#Slice the data, find qubit freq and linewidth
for idx in range(len(pump_power)):
    to_plot[idx, :] = np.real(IQ_rotate(signal[idx,:]))
    to_plot[idx,:] = abs(to_plot[idx,:] - np.mean(to_plot[idx,:]))
#     #Fit Lorentzian and Gaussian
    freq_guess = qubit_freq[np.argmax(to_plot[idx,:])]
    guess_lorentzian = ([np.max(to_plot[idx, :]) - np.min(to_plot[idx, :]), freq_guess, 2e6,0])
    opt_lorentzian, cov_lorentzian = curve_fit(lorentzian, xdata=qubit_freq, ydata=to_plot[idx,:],p0=guess_lorentzian)
    err_lorentzian = np.sqrt(np.diag(cov_lorentzian))
    guess_gaussian = ([np.max(to_plot[idx, :]) - np.min(to_plot[idx, :]), freq_guess, 2e6,0])
    opt_gaussian, cov_gaussian = curve_fit(gaussian_func, xdata=qubit_freq, ydata=to_plot[idx, :], p0=guess_gaussian)
    err_gaussian = np.sqrt(np.diag(cov_gaussian))
    if abs(err_gaussian[2]) > abs(err_lorentzian[2]):
        freq[idx] = opt_lorentzian[1]
        freq_err[idx] = err_lorentzian[1]
        width[idx] = opt_lorentzian[2]
        width_err[idx] = err_lorentzian[2]
    else:
        freq[idx] = opt_gaussian[1]
        freq_err[idx] = err_gaussian[1]
        width[idx] = opt_gaussian[2]
        width_err[idx] = err_gaussian[2]

#Convert, dBm = 10log10(P/1mW)
power_W = 10**(pump_power/10.0)*1e-3 # in W

#3D
plt.figure(1)
X,Y = np.meshgrid(pump_power, qubit_freq/1e9)
Z = to_plot.transpose()
plt.pcolormesh(X,Y,Z, cmap = 'GnBu', vmin = 0, vmax = 0.0003)
# plt.colorbar()
plt.tick_params(labelsize = 16.0)
plt.ylim(5.95,6.1)
plt.ylabel('Qubit tone frequency (MHz)', size = 14.0)
plt.xlabel('ac Stark tone power (uW)', size = 14.0)


# Plot frequency shift
plt.figure(2)
plt.errorbar(power_W*1e6, (-freq+freq[0])*1e-6, yerr=freq_err*1e-6, linestyle='none', marker='d', mfc='none', ms=5,
                 mew=2)
plt.tick_params(labelsize = 16.0)
plt.ylabel('Frequency shift (MHz)', size = 14.0)
plt.xlabel('ac Stark tone power (uW)', size = 14.0)
#fit
guess = ([2e8, 60e6])
opt, cov = curve_fit(acStarkShift,ydata =-freq+freq[0], xdata = power_W, p0=guess)
plt.plot(power_W*1e6, acStarkShift(power_W,*opt)*1e-6, linewidth = 2.0)
# print (opt)


#Plot spectroscopic width
plt.figure(3)
plt.errorbar(power_W*1e6, width*1e-6, yerr=width_err*1e-6, linestyle='none', marker='d', mfc='none', ms=5,
                 mew=2)
plt.tick_params(labelsize = 16.0)
plt.ylabel('Linewidth (MHz)', size = 14.0)
plt.xlabel('ac Stark tone power (uW)', size = 14.0)
# fit
# guess = ([2e8, (4e-6)**-1, 60e6])
# opt, cov = curve_fit(acStarkDephasing,ydata =width, xdata = power_W, p0=guess)
# plt.plot(power_W*1e6, acStarkDephasing(power_W,*opt)*1e-6, linewidth = 2.0)

plt.show()