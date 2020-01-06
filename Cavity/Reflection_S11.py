import numpy
from matplotlib import pyplot as plt

plt.close('all')
f, axarr = plt.subplots(2, sharex=True)

fo = 10.6e9
def S_11(f,q_ext1,q_ext2,q_int1,q_int2):
    q_ext = q_ext1 + 1j*q_ext2
    q_int = q_int1 + 1j*q_int2
    S = (2j*(f-fo)/fo - (q_ext)**(-1)  + q_int**(-1))/(2j*(f-fo)/fo + (q_ext)**(-1) + q_int**(-1))
    return S

f = linspace(10.5999e9,10.6001e9,10000)
qint = linspace (1,100,9)

S = S_11(f,4e6,0,4.2e6,0)
S_mag = 20*log10(abs(S))
S_phase = unwrap(angle(S))
axarr[0].plot(f,S_mag)
axarr[1].plot(f,S_phase)

S = S_11(f,4e6,0,5e6,0)
S_mag = 20*log10(abs(S))
S_phase = unwrap(angle(S))
axarr[0].plot(f,S_mag)
axarr[1].plot(f,S_phase)