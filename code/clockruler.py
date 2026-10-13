"""Порядок как часы и как линейка: что он знает и чего не знает.

Теорема вечера (boost.py): пространственный поперечник не функция порядка.
Но собственное время — обязано быть: |I| ~ rho*(pi/24)*tau^4.
Мерим оба утверждения на одних данных.
"""
import numpy as np
from particles import diamond, rel_of

N = 2500
rs = np.random.default_rng(4242)
X = diamond(N, rs)
R = rel_of(X)
A = R.astype(np.float32)
INT = (A @ A)                      # INT[i,j] = |I(i,j)| при R[i,j]
i, j = np.nonzero(R)
pick = rs.choice(len(i), 60000, replace=False)
i, j = i[pick], j[pick]
n = INT[i, j]
dt = X[j, 0] - X[i, 0]
dx = np.linalg.norm(X[j, 1:] - X[i, 1:], axis=1)
tau = np.sqrt(np.maximum(dt**2 - dx**2, 0))
rho = N / (np.pi/24)               # ромб T=1: V = pi/24
tau_ord = (24.0 * np.maximum(n, 0) / (np.pi * rho))**0.25

print(f'{len(i)} связанных пар, N = {N}')
print()
print('ЧАСЫ: порядковое собственное время против истинного')
print(f'  корреляция |I|^(1/4) с tau:          '
      f'{np.corrcoef(tau_ord, tau)[0,1]:+.4f}')
print(f'  корреляция (все пары, log):          '
      f'{np.corrcoef(np.log(tau_ord+1e-9), np.log(tau+1e-9))[0,1]:+.4f}')
print()
print('ЛИНЕЙКА: знает ли порядок пространственное расстояние')
print(f'  корреляция |I| с |dx|:               '
      f'{np.corrcoef(n, dx)[0,1]:+.4f}')
print()
print('  при ФИКСИРОВАННОМ |I| — что остаётся от |dx|:')
print(f"{'|I|':>8}{'пар':>8}{'tau средн':>11}{'tau ст.откл':>13}"
      f"{'|dx| средн':>12}{'|dx| ст.откл':>14}{'|dx| разброс':>14}")
for lo, hi in ((1,2),(3,5),(8,12),(20,30),(50,80),(150,250)):
    m = (n >= lo) & (n <= hi)
    if m.sum() < 50: continue
    t_, x_ = tau[m], dx[m]
    print(f'{f"{lo}-{hi}":>8}{int(m.sum()):>8}{t_.mean():>11.4f}'
          f'{t_.std():>13.4f}{x_.mean():>12.4f}{x_.std():>14.4f}'
          f'{x_.std()/x_.mean():>14.4f}')
print()
print('  tau при фиксированном |I| узко => ЧАСЫ; |dx| широко => СЛЕПАЯ ЛИНЕЙКА')
