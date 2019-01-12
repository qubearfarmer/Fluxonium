import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

cmin = -1
cmax = 1

def lorentzian (x,gamma,x0,h,offset):
    return offset+(h*(gamma/2)**2)/((x-x0)**2+(gamma/2)**2)
fit_freq = []
fit_width =[]
power_all = []

#File path
# directory = 'G:\Projects\Fluxonium\Data\Fluxonium #28\Two_tone'
# measurement = '043018_Two_tone_power_YOKO_5to-30dBm_Cav_7.33756GHz&1.23mA_QuBit0.2to0.35GHz&25dBm'
# path = directory + '\\' + measurement
#
# #Read data
# power = np.genfromtxt(path + '_POWER.csv')
# power = power[1:-1]
# freq = np.genfromtxt(path + '_FREQ.csv')
# freq = freq[1::]
# data = np.genfromtxt(path + '_PHASEMAG.csv')
# phase = data[1::,0] #phase is recorded in rad
# mag = data[1::,1]
# magdB = 10*np.log10(mag)
# Z = np.zeros((len(power), len(freq)))
# # print (phase)
# for idx in range(len(power)):
#     temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
#     temp = temp*180/(np.pi)
#     # temp = mag[idx*len(freq):(idx+1)*len(freq)]
#     Z[idx,:] = temp - np.mean(temp)
#
# plt.figure(1)
# X,Y = np.meshgrid(power, freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu')#, vmin = cmin, vmax = cmax)
# plt.ylim([freq[0],freq[-1]])
#
# plt.figure(2)
# X,Y = np.meshgrid(np.power(10,power/10.0), freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu')#, vmin = cmin, vmax = cmax)
###########################################################################################
directory = 'G:\Projects\Fluxonium\Data\Julius II\Two_tone'
measurement = 'Two_tone_spec_POWER_-30to-10mA_Cav_7.507GHz_QuBit5.3to5.5GHz'
path = directory + '\\' + measurement

#Read data
power = np.genfromtxt(path + '_POWER.csv')
power = power[1::]
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1::,0] #phase is recorded in rad
mag = data[1::,1]
magdB = 10*np.log10(mag)
Z = np.zeros((len(power), len(freq)))
# print (phase)
for idx in range(len(power)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
    fo_guess = np.max(Z[idx,:])
    guess = ([0.001, freq[np.argmax(temp)], np.max(temp) - np.min(temp), np.min(temp)])
    popt, pcov = curve_fit(lorentzian, freq, temp, p0=guess)
    fit_freq = np.append(fit_freq, popt[1])
    fit_width = np.append(fit_width, popt[0])

plt.figure(1)
X,Y = np.meshgrid(power, freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu')#, vmin = cmin, vmax = cmax)


# plt.figure(2)
# X,Y = np.meshgrid(np.power(10,power/10.0), freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu')#, vmin = cmin, vmax = cmax)

power_all = np.append(power_all, power)


measurement = 'Two_tone_spec_POWER_-9to-11mA_Cav_7.507GHz_QuBit5.25to5.5GHz'
path = directory + '\\' + measurement

#Read data
power = np.genfromtxt(path + '_POWER.csv')
power = power[1::]
power = power + (np.max(power)-np.min(power))/(len(power)-1)
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1::,0] #phase is recorded in rad
mag = data[1::,1]
magdB = 10*np.log10(mag)
Z = np.zeros((len(power), len(freq)))
# print (phase)
for idx in range(len(power)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
    fo_guess = np.max(Z[idx, :])
    guess = ([0.01, freq[np.argmax(temp)], np.max(temp) - np.min(temp), np.min(temp)])
    popt, pcov = curve_fit(lorentzian, freq, temp, p0=guess)
    fit_freq = np.append(fit_freq, popt[1])
    fit_width = np.append(fit_width, popt[0])

plt.figure(1)
X,Y = np.meshgrid(power, freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu')#, vmin = cmin, vmax = cmax)


# plt.figure(2)
# X,Y = np.meshgrid(np.power(10,power/10.0), freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu')#, vmin = cmin, vmax = cmax)

power_all = np.append(power_all, power-(np.max(power)-np.min(power))/(len(power)-1))


measurement = 'Two_tone_spec_POWER_-10to0mA_Cav_7.507GHz_QuBit5.0to5.4GHz'
path = directory + '\\' + measurement

#Read data
power = np.genfromtxt(path + '_POWER.csv')
power = power[1:-1]
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1::,0] #phase is recorded in rad
mag = data[1::,1]
magdB = 10*np.log10(mag)
Z = np.zeros((len(power), len(freq)))
# print (phase)
for idx in range(len(power)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
    fo_guess = np.max(Z[idx, :])
    guess = ([0.01, freq[np.argmax(temp)], np.max(temp) - np.min(temp), np.min(temp)])
    popt, pcov = curve_fit(lorentzian, freq, temp, p0=guess)
    fit_freq = np.append(fit_freq, popt[1])
    fit_width = np.append(fit_width, popt[0])

plt.figure(1)
X,Y = np.meshgrid(power, freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu')#, vmin = cmin, vmax = cmax)


# plt.figure(2)
# X,Y = np.meshgrid(np.power(10,power/10.0), freq)
# plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu')#, vmin = cmin, vmax = cmax)

power_all = np.append(power_all, power)

plt.figure(1)
plt.plot(power_all, fit_freq, 'r.')

plt.figure(3)
plt.plot(power_all, fit_width*1e3, 'b.')
# plt.xlabel('Power (dBm)')
# plt.ylabel('Linewidth (MHz)')
###########################################
h = 6.626e-34
f_cav = 7.12e9
kappa = 15e6*2*np.pi
attenuation_approx = 80 #dB
f_bare = fit_freq[0]*1e9
chi_approx = 0.036e6*2*np.pi
def ac_stark(power_dBm, attenuation, fo):
    power_W = 10**((power_dBm-attenuation)/10.0)*1e-3
    n = power_W/(h*f_cav) * (kappa/2)/((kappa/2)**2+chi_approx**2)
    f = fo - 2.0*n*chi_approx
    return f
guess = ([attenuation_approx, f_bare])
opt,cov = curve_fit(ac_stark,power_all, fit_freq*1e9, guess)
attenuation_fit, f_fit = opt
plt.figure(1)
power_nice = np.linspace(-30, 0, 151)
freq_fit = ac_stark(power_nice, attenuation_fit, f_fit)
plt.plot(power_nice,freq_fit/1e9, linewidth = 3, linestyle = '--', color = 'magenta')
print ("Attenuation = " + str(attenuation_fit) +'dB' + '\n'
       +"Qubit bare freq = " + str(f_fit/1e9) +'GHz' + '\n'
       +"chi  = " + str(chi_approx/1e6) + 'MHz')
###########################################
power_W = 10**((0-attenuation_fit)/10.0)*1e-3
n = power_W/(h*f_cav) * (kappa/2)/((kappa/2)**2+chi_approx**2)
print (n)


plt.show()