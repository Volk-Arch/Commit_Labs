"""Фаза шага НАПРЯМУЮ, без гауссова приближения, плюс цена малого eps."""
import math
import numpy as np
from weight import diamond, cone_rel, between, per_event

RHO = 150.0*24/np.pi
print('ФАЗА ШАГА НАПРЯМУЮ: |<exp(i dS)>| по событиям верхней половины')
print('рядом — гауссова оценка exp(-s^2/2) и дальнобойность веса')
print()
print(f"{'eps':>6}{'T':>5}{'N':>7}{'разброс':>9}{'фаза прямо':>12}"
      f"{'гаусс':>9}{'избыт.куртозис':>16}{'reach/мир':>11}")
for eps in (0.05, 0.1, 0.2):
    for T in (1.5, 2.0, 2.5):
        ph, gs, sd, ku = [], [], [], []
        for t in range(4):
            rs = np.random.default_rng(880+17*t)
            X = diamond(T, rs); X = X[np.argsort(X[:,0])]
            R = cone_rel(X); btw = between(R)
            s,_ = per_event(R, btw, eps); up = s[len(X)//2:]
            ph.append(abs(np.exp(1j*up).mean()))
            sd.append(up.std()); gs.append(math.exp(-up.var()/2))
            z = (up-up.mean())/up.std()
            ku.append((z**4).mean()-3)
            N = len(X); del R, btw, X
        # дальнобойность: носитель веса по n ~ 1/eps -> tau
        nmax = 4.0/eps
        reach = (nmax*24/(RHO*np.pi))**0.25
        print(f'{eps:>6.2f}{T:>5.1f}{N:>7}{np.mean(sd):>9.4f}'
              f'{np.mean(ph):>12.4f}{np.mean(gs):>9.4f}'
              f'{np.mean(ku):>16.1f}{reach/T:>11.2f}')
    print()
