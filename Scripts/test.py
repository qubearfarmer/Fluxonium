# libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("white")
import pandas as pd

kappa = 3.5e6
chi = 5e6
n_eff = np.linspace(0,100,10001)
gamma_0 = 1/4e-6
def gamma(kappa, chi, n_eff):
    return kappa/2 * np.real(np.sqrt((1+1j*chi/kappa)**2 + 4j*chi*n_eff/kappa)-1) + gamma_0

plt.plot(n_eff,gamma(kappa, chi, n_eff))
plt.show()

