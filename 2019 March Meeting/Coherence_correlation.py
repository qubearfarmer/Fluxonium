import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

latex_table = """
\begin{tabular}{||c| c c c c c c c c c c c c c ||}
 \hline
 Qubit &$E_J$ & $E_C$ & $E_L$ & $N$ & $T_1$ & $T_2$ & $\omega_{01}/2\pi$ & $\omega_{12}/\omega_{01}$ & $\chi_{01}/2\pi$ & $\tan \delta_C$ & $\tan \delta_{\mathrm{AlOx}}$ &  $x$   &$\tan \delta_L$ \\ 
 \hline
  & GHz & GHz & GHz & - & $\mathrm{\mu s}$ & $\mathrm{\mu s}$ & GHz & - & MHz & $\times 10^{-6}$ & $\times 10^{-4}$ & $\times 10^{-8}$ & $\times 10^{-8}$\\
 \hline\hline
 A & 3 & 0.84 & 1 & 100 & 110 & 160 & 0.78& 3.4 & 0.27 & 1.7 &1.1 & 3.84  &  15.4\\ 
 \hline
 B & 4.86 &0.84 & 1.14 & 136 & 250 & 150 & 0.32& 11.1 & 0.57 & 1.5 &1.3& 0.52 & 2.03  \\
 \hline
 C & $2.2^*$ & 0.55 & 0.72 & 102 & 260 & 350 & 0.48& 3.8 & 0.08  & 1.15 &0.9& 1.77 & 5.75  \\
 \hline
 D & 2.2 & 0.83 & 0.52 & 196 & 70 & 90 & 0.56& 4.1 & 0.1 & 1.9 &4.0 & 7 &28.25 \\
 \hline
 E & 1.6 & 0.86 & 0.5 & 100 & 108 & 140 & 0.83& 2.5 & 0.05 & 3.25 &1.0& 7.8 & 30.22 \\
 \hline
 F & 3.4 & 0.8 & 0.41 & 348 & 270 & 165 & 0.17& 18.3 & 0.28  & 0.3 &4.5 & 0.63 & 2.1 \\
 \hline
 G & 1.65 & 1.14 & 0.19 & 400 & 110 & 140 & 0.55& 4.1 & 0.03 & 5.6 &3.8  & 8.65 & 34.9  \\
\hline
 H & 4.43 & 1 & 0.79 & 100 & 230 & 235 & 0.32& 11.8 & 0.1 & 1.1&0.9 & 0.72  & 2.85 \\
\hline
\end{tabular}
"""

table = latex_table.replace('\times 10^', 'e').replace('mathrm', '').replace('mu', 'u')
table = table.replace('\\', '').replace('$', '').replace('{', '').replace('}', '')
table = table.replace('\tan', 'tan_').replace(' ', '').replace('^*', '')
table = [line for line in table.split('\n') if line.find('&') >= 0]
names = table[0].split('&')
units = table[1].split('&')
columns=[('%s (%s)' % (name, unit)).replace(' (-)', '').replace(' ()', '')
         for name, unit in zip(names, units)]
df = pd.DataFrame(columns=columns)
df = pd.concat([pd.DataFrame([table[i].split('&')], columns=columns)
                for i in range(2, len(table))],
                ignore_index=True)
df[columns[1:]] = df[columns[1:]].apply(pd.to_numeric, errors='raise')
df = df.set_index('Qubit')
df.head(100)

df.describe()

sns.heatmap(df.corr(),
            annot=True,
            fmt=".2f",
            square=True,
            cmap='bwr',
            vmin=-1.,
            vmax=1.)
plt.gcf().set_size_inches(10, 7.5)
plt.show()