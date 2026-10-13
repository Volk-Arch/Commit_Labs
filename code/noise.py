"""Как растёт шум поля БД с измельчением — и сколько нужно сглаживания.

Сигнал: <b> = -(sqrt6/8) ell^2 R  ~  N^(-1/2).
Шум: sigma(b) — мерим.
Отношение сигнал/шум на событие и нужное n для 3 сигм при R = 1.
"""
import numpy as np, math
from particles import diamond, rel_of
from bdfield import bd_density, heights

print(f"{'N':>7}{'ell':>9}{'ell^2':>9}{'sigma(b)':>11}{'<b> глуб':>11}"
      f"{'сигнал R=1':>12}{'с/ш на соб.':>13}{'n для 3 сигм':>15}")
rows = []
for N in (600, 1200, 2400, 4800):
    ell = (math.pi/24.0/N)**0.25
    sg, mb = [], []
    for s in range(6):
        X = diamond(N, np.random.default_rng(3700+29*s))
        R = rel_of(X); b = bd_density(R); h = heights(R)
        hm = int(h.max()); deep = (h >= 3) & (h <= hm-2)
        if deep.sum() < 50: continue
        sg.append(b[deep].std()); mb.append(b[deep].mean())
        del R, X
    s_ = float(np.mean(sg)); m_ = float(np.mean(mb))
    sig = 0.306 * ell**2            # |<b>| при R = 1
    need = (3.0*s_/sig)**2
    rows.append((N, ell, s_, sig))
    print(f'{N:>7}{ell:>9.4f}{ell**2:>9.5f}{s_:>11.1f}{m_:>+11.2f}'
          f'{sig:>12.5f}{sig/s_:>13.2e}{need:>15.1e}')

A = np.array(rows)
f = lambda y: np.polyfit(np.log(A[:,0]), np.log(y), 1)[0]
print()
print(f'  шум sigma(b)   ~ N^{f(A[:,2]):+.3f}')
print(f'  сигнал (R=1)   ~ N^{f(A[:,3]):+.3f}   (ожидание -0.50, ибо ell^2)')
print(f'  с/ш на событие ~ N^{f(A[:,3]/A[:,2]):+.3f}  — ухудшается с N')
print()
print('  СГЛАЖИВАНИЕ: оператор с промежуточным масштабом ell_k = k*ell')
print('  усредняет по ~k^4 событиям, то есть шум падает как k^-2,')
print('  а сигнал сохраняется. Нужное k для 3 сигм при R=1 и n событиях:')
print(f"{'N':>7}{'n глубины':>12}{'нужное k':>11}{'ell_k/ell':>12}")
for N, ell, s_, sig in rows:
    n = 0.6*N
    k = math.sqrt(3.0*s_/(sig*math.sqrt(n)))
    print(f'{N:>7}{int(n):>12}{k:>11.1f}{k:>12.1f}')
