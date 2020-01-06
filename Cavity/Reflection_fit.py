from scipy import *
from scipy.optimize import curve_fit
import numpy
from matplotlib import pyplot as plt
from matplotlib import style

style.use('seaborn-paper')
plt.close('all')
# f, axarr = plt.subplots(2, sharex=True)
# Specify the path to the file. Assign frequency column and amplitude column
# File has .csv format. First column specifies frequency
# The path to the file has double '\' before a number

path1 = 'Z:\Projects\Cavity and Wave Guide\Copper cavity\\2018_07_30_7p5GHz_narrow\\readout_Mag.csv'
path2 = 'Z:\Projects\Cavity and Wave Guide\Copper cavity\\2018_07_30_7p5GHz_narrow\\readout_phase.csv'
data1 = numpy.genfromtxt(path1, skip_header=3, delimiter=',')  # Magnitude in dB
freq = data1[:, 0] / 1e9
mag_data = data1[:, 1]

data2 = numpy.genfromtxt(path2, skip_header=3, delimiter=',')  # Phase in degrees
phase_data = data2[:, 1]

# Find resonant frequency and offsets
index = 0
fo = 0
amp_cal = 0
indexo = 0
phase_cal = (phase_data[0] + phase_data[len(phase_data) - 1]) / 2.0
for amp in mag_data:
    if amp_cal > amp:
        amp_cal = amp
        indexo = index
    index = index + 1
# for amp in phase_data:
#    if abs(amp - phase_cal) < 2:
#        indexo=index
#    index = index+1
fo = freq[indexo]
# offset=phase_data[indexo]
# phase_data = phase_data - offset
# mag_data = mag_data - mag_data[0]

# Convert to appropriate units
mag_dataN = 10.0 ** (mag_data / 20.0)
phase_dataN = phase_data * pi / 180.0

# If real and imaginary data are not available, compute them here. Skip if data are available
real_data = mag_dataN * cos(phase_dataN)
imag_data = mag_dataN * sin(phase_dataN)
all_data = numpy.array([real_data, imag_data]).flatten()


# Theoretical model

def S_11(f, q_ext1, q_ext2, q_int, tau1, tau2, tau3):
    S = (2j * (f - fo) / fo - (q_ext1 + 1j * q_ext2) ** (-1) + q_int ** (-1)) / (
                2j * (f - fo) / fo + (q_ext1 + 1j * q_ext2) ** (-1) + q_int ** (-1))
    S = S * tau1 * exp(1j * (tau2 + f * tau3))
    return numpy.array([S.real, S.imag]).flatten()


# Make a guess here for the values of Q_ext, Q_int,
# tau1 (magnitude offset), tau2 (linear phase offset), tau3 (freq dependent phase offset)

guess = ([0.5e3, 1000, 5e3, 0, 0, 0])

qopt, qcov = curve_fit(S_11, freq, all_data, guess)
q_external1 = qopt[0]
q_external2 = qopt[1]
q_internal = qopt[2]
tau_final1 = qopt[3]
tau_final2 = qopt[4]
tau_final3 = qopt[5]
error = sqrt(diag(qcov))[2]

# Check

S_final = (2j * (freq - fo) / fo - (q_external1 + 1j * q_external2) ** (-1) + q_internal ** (-1)) / (
            2j * (freq - fo) / fo + (q_external1 + 1j * q_external2) ** (-1) + q_internal ** (-1))
S_final = S_final * tau_final1 * exp(1j * (tau_final2 + freq * tau_final3))
mag_final = abs(S_final)
phase_final = angle(S_final)
phase_final = unwrap(phase_final)
real_final = S_final.real
imag_final = S_final.imag

plt.figure(1)

plt.errorbar(freq,  mag_dataN, fmt='s', mfc='none', mew=0.5)
plt.plot(freq, mag_final, linewidth = 4.0)
plt.figure(2)
plt.errorbar(freq,  phase_dataN, fmt='s', mfc='none', mew=0.5)
plt.plot(freq, phase_final, linewidth = 4.0)


print('Q_ext= ' + str(qopt[0]) + ", Q_int= " + str(qopt[2]) + ", f_o= " + str(fo) + " , error= " + str(error))
plt.show()