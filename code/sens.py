"""Стоит ли большой прогон: заметен ли объект для САМОГО действия.

Если трубка (или дырка) меняет S меньше, чем разброс S по засевам, то
exp(-beta S) про объект знать не может ни при какой выборке.
Это та же логика, что в bdwall-doc.md: сначала сила замера, потом вывод.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from ens2 import action_and_pairs, sprinkle

print(f"{'N':>6}{'ell':>8}{'⟨S⟩ пусто':>12}{'разброс':>10}"
      f"{'ΔS трубка':>12}{'ΔS дырка':>11}{'σ/|ΔS| труб':>13}"
      f"{'засевов для 3σ':>16}")
for N in (220, 440, 880):
    ell = (math.pi/24.0/N)**0.25
    S0, St, Sh = [], [], []
    for s in range(14):
        rs = np.random.default_rng(600+13*s)
        X = sprinkle(N, 1.0, rs, 0.0, [])
        S0.append(action_and_pairs(X)[0])
        frz = np.zeros((26,4)); frz[:,0] = np.linspace(0.12,0.88,26)
        St.append(action_and_pairs(np.vstack([X, frz]))[0])
        rs2 = np.random.default_rng(600+13*s)
        Xh = sprinkle(N, 1.0, rs2, 0.16, [np.zeros(3)])
        Sh.append(action_and_pairs(Xh)[0])
    S0 = np.array(S0); St = np.array(St); Sh = np.array(Sh)
    dt_ = (St-S0).mean(); dh = (Sh-S0).mean()
    sd = S0.std(ddof=1)
    need = (3.0*sd/abs(dt_))**2 if abs(dt_) > 1e-9 else float('inf')
    print(f'{N:>6}{ell:>8.4f}{S0.mean():>12.1f}{sd:>10.1f}'
          f'{dt_:>+12.1f}{dh:>+11.1f}{sd/max(abs(dt_),1e-9):>13.2f}'
          f'{need:>16.1f}')
print()
print('  ΔS от объекта МЕНЬШЕ разброса => вес объекта не видит;')
print('  БОЛЬШЕ => видит, и дело в перемешивании, а не в силе')
