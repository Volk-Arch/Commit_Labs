"""Рост, подгоняющий перекрытие прошлых под цель.

Не минимизация (она ломала воспроизводимость через крайнюю статистику),
а ПОДГОНКА: из K кандидатов берётся тот, чьё среднее перекрытие с
прошлыми существующих событий ближе к целевому J*. У подгонки нет
убегания, поэтому воспроизводимость может сохраниться.
"""
import math, numpy as np
from core import mm_dim, height, spectral, frac_pairs, links_of

def grow_j(N, Jstar, p, rs, K=16, alpha=1.0):
    R = np.zeros((N,N), bool); desc = np.zeros(N, np.int64)
    for k in range(1, N):
        w = (desc[:k].astype(np.float64)+1.0)**alpha; w /= w.sum()
        m = min(max(1,int(p*k)), k)
        S = np.zeros((k,K), bool)
        for j in range(K):
            S[rs.choice(k, m, replace=False, p=w), j] = True
        A = R[:k,:k].astype(np.float32)
        D = S | ((A.T @ S.astype(np.float32)) > 0)
        if k > 12:
            Df = D.astype(np.float32)
            inter = A @ Df                       # |past(i) & D_j|
            szP = A.sum(1); szD = Df.sum(0)
            union = szP[:,None] + szD[None,:] - inter
            J = np.where(union>0, inter/np.maximum(union,1e-9), 0.0).mean(0)
            j = int(np.argmin(np.abs(J - Jstar)))
        else:
            j = int(rs.integers(K))
        d = D[:,j]; R[k,:k] = d; desc[:k] += d
    return R

S = 6
print(f'подгонка перекрытия, N=600, засевов {S}, alpha=1, p=0.05')
print('ЦЕЛЬ: доля ~0.10, разм. 4.00, спектр +0.010 при с.к.о. ~0.003')
print('эталон конуса (N=700): J связ. 0.129, J несравн. 0.169')
print()
print(f"{'J*':>6}{'доля':>9}{'разм.':>7}{'высота':>7}{'<спектр>':>11}"
      f"{'с.к.о.':>9}{'связ/соб':>9}")
for Js in (0.06, 0.10, 0.14, 0.18, 0.25):
    v,f,d,h,l = [],[],[],[],[]
    for t in range(S):
        rs = np.random.default_rng(71000+977*t)
        R = grow_j(600, Js, 0.05, rs); ff = frac_pairs(R)
        v.append(spectral(R)); f.append(ff); d.append(mm_dim(ff))
        h.append(height(R)[0]); l.append(links_of(R).sum()/600); del R
    v = np.array(v)
    print(f'{Js:>6.2f}{np.mean(f):>9.4f}{np.mean(d):>7.2f}{np.mean(h):>7.1f}'
          f'{v.mean():>+11.4f}{v.std():>9.4f}{np.mean(l):>9.1f}', flush=True)
