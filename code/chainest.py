"""Насколько точно длина цепи оценивает собственное время.

L ~ m_4 * rho^(1/4) * tau. Вопрос в разбросе: Грам строится из разностей
квадратов L, значит относительная ошибка в L входит удвоенной и ещё
усиливается сокращением.
"""
import numpy as np
from gps import slab, cone_rel, longest_all

N, T, Lx = 6000, 4.0, 0.5
rho = N/(T*(2*Lx)**3)
rs = np.random.default_rng(5)
X = slab(N, T, rs, Lx); R = cone_rel(X)
srcs = np.flatnonzero(X[:,0] < 0.1*T)[:12]
Lc = longest_all(R, srcs)
print(f'N={N}, плотность {rho:.0f}, rho^(1/4)={rho**0.25:.2f}')
print('ожидание: L ~ m*rho^(1/4)*tau, m_4 ~ 2.0  =>  L ~ '
      f'{2*rho**0.25:.1f} * tau')
print()
print(f"{'tau':>7}{'пар':>7}{'<L>':>8}{'с.к.о.':>8}{'отн.разброс':>13}"
      f"{'L/tau':>8}{'разброс (L^2)':>15}")
s = 0
src = int(srcs[s])
tau_all, L_all = [], []
for b in range(len(X)):
    if Lc[b, s] > 0:
        dt = X[b,0]-X[src,0]; dx = np.linalg.norm(X[b,1:]-X[src,1:])
        if dt > dx:
            tau_all.append(np.sqrt(dt*dt-dx*dx)); L_all.append(Lc[b,s])
tau_all = np.array(tau_all); L_all = np.array(L_all, float)
for lo, hi in ((0.0,0.2),(0.2,0.4),(0.4,0.7),(0.7,1.1),(1.1,1.6),(1.6,2.5)):
    m = (tau_all>=lo)&(tau_all<hi)
    if m.sum() < 20: continue
    L = L_all[m]; tt = tau_all[m].mean()
    rel = L.std()/L.mean()
    # относительный разброс квадрата примерно вдвое больше
    print(f'{tt:>7.2f}{int(m.sum()):>7}{L.mean():>8.1f}{L.std():>8.2f}'
          f'{rel:>13.3f}{L.mean()/tt:>8.2f}{2*rel:>15.3f}')
print()
print('ЧТОБЫ сигнатура Грама была надёжна, нужен относительный разброс')
print('квадрата много меньше относительных РАЗНОСТЕЙ между его членами.')
print('При разбросе 2*rel ~ 0.3 и сокращении на порядок — безнадёжно.')
