import numpy as np
from matplotlib import pyplot as plt
from qutip import*

state0 = basis(2,0)
state1 = basis(2,1)
print state0, state1
wa = 10
g = 0.1
w = 10
Ha = (-wa/2-w)*state0*state0.dag() + (wa/2-w)*state1*state1.dag()
Hc = g*(state0*state1.dag()+state1*state0.dag())
H = Ha + Hc
print H
print sigmax()
# H = g*sigmax()
time_list = np.linspace(0,100,101)
psi0 = state0
collapse_ops = []
ope0 = state0*state0.dag()
ope1 = state1*state1.dag()
# print ope0, ope1
quantum_operators = [ope0, ope1]
# quantum_operators = sigmaz()
output = mesolve(H, psi0, time_list, collapse_ops, quantum_operators)
st0 = output.expect[0]
# st1 = output.expect[1]

plt.plot(time_list, st0)
plt.show()