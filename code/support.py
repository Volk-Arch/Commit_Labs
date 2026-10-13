"""Носитель БД: точен ли обрыв, и как далеко он достаёт В ПРОСТРАНСТВЕ.

Проверяем три вещи:
 1. ровно ли нуль за обрывом (не «мало», а ровно);
 2. размер носителя в длинах дискретности — должен стоять при росте N;
 3. пространственный разброс влияющих событий — стоит или сжимается.
"""
import numpy as np
from particles import diamond, rel_of

W = np.array([-1.0, 9.0, -16.0, 8.0])
K = np.array([W[0] if 0 else 0] * 0)
KB = np.array([(W[m-1] if 1 <= m <= 4 else 0.0) -
               (W[m] if 0 <= m <= 3 else 0.0) for m in range(5)])
print('коэффициенты при c_m:', KB)
print()
print(f"{'N':>7}{'ell':>8}{'носитель tau':>14}{'в ell':>8}"
      f"{'ровно 0 за ним':>16}{'|dx| влияющих':>15}{'в ell':>8}"
      f"{'|dx|/радиус мира':>18}")
for N in (800, 1600, 3200):
    ell = (np.pi/24.0/N)**0.25
    sup, exact, dxm, frac = [], [], [], []
    for s in range(3):
        X = diamond(N, np.random.default_rng(3300+41*s))
        R = rel_of(X); A = R.astype(np.float32); INT = A @ A
        cand = np.flatnonzero(R.sum(0) > 60)
        if len(cand) < 3: continue
        xs = np.random.default_rng(9+s).choice(cand, min(20,len(cand)),
                                               replace=False)
        for x in xs:
            x = int(x); past = np.flatnonzero(R[:, x])
            if len(past) < 40: continue
            nv = INT[:, x]
            C = np.stack([A.T @ (nv == m).astype(np.float32)
                          for m in range(5)])
            nz = nv[past].astype(np.int64)
            wz = np.where(nz <= 3, W[np.clip(nz,0,3)], 0.0)
            db = (KB[:,None]*C[:,past]).sum(0) - wz
            dt = X[x,0]-X[past,0]
            dxv = np.linalg.norm(X[x,1:]-X[past,1:], axis=1)
            tau = np.sqrt(np.maximum(dt**2-dxv**2, 0))
            nzm = np.abs(db) > 1e-9
            if nzm.sum() < 5: continue
            sup.append(tau[nzm].max())
            far = tau > tau[nzm].max()*1.05
            exact.append(float(np.abs(db[far]).max()) if far.any() else 0.0)
            dxm.append(dxv[nzm].mean())
            frac.append(dxv[nzm].mean())
        del R, A, INT, X
    print(f'{N:>7}{ell:>8.4f}{np.mean(sup):>14.4f}'
          f'{np.mean(sup)/ell:>8.2f}{np.max(exact):>16.1e}'
          f'{np.mean(dxm):>15.4f}{np.mean(dxm)/ell:>8.2f}'
          f'{np.mean(frac)/1.0:>18.4f}')
print()
print('  носитель в ell стоит => локальность настоящая')
print('  |dx|/радиус мира стоит => влияющие события достают пол-мира всегда')
