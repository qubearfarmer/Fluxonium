import numpy as np
from matplotlib import pyplot as plt

def reflection_signal(freq, freq_res, Q_ext, Q_int):
    numerator = 2j*(freq-freq_res)/freq_res - Q_ext**-1 + Q_int**-1
    denominator = 2j*(freq-freq_res)/freq_res + Q_ext**-1 + Q_int**-1
    s11 = numerator / denominator
    return s11

freq = np.linspace(9.8,10.2,201)*1e9
freq_res = 10e9

Q_ext = 1000
Q_int = 10000
ref_real = np.real(reflection_signal(freq, freq_res, Q_ext, Q_int))
ref_imag = np.imag(reflection_signal(freq, freq_res, Q_ext, Q_int))
plt.plot(freq/1e9, ref_real**2 + ref_imag**2)
plt.show()