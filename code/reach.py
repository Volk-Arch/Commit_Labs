"""Какое перекрытие вообще достижимо в пуле кандидатов."""
import numpy as np
from core import frac_pairs
import growth2
from growth2 import grow

def probe(N, p, alpha, K=64):
    growth2._ALPHA = alpha
    rs = np.random.default_rng(81000)
    R,_ = grow(N, 'alpha', p, 0.0, 0.2, 8, rs)   # вырастить мир
    k = N
    A = R.astype(np.float32)
    # пул кандидатов на следующем шаге
    w = np.ones(k)/k
    out = []
    for m_frac in (0.01, 0.03, 0.05, 0.10, 0.20):
        m = max(1, int(m_frac*k))
        S = np.zeros((k,K), bool)
        for j in range(K):
            S[rs.choice(k, m, replace=False), j] = True
        D = S | ((A.T @ S.astype(np.float32)) > 0)
        Df = D.astype(np.float32)
        inter = A @ Df; szP = A.sum(1); szD = Df.sum(0)
        union = szP[:,None]+szD[None,:]-inter
        J = np.where(union>0, inter/np.maximum(union,1e-9), 0.0).mean(0)
        out.append((m_frac, szD.mean()/k, J.min(), J.mean(), J.max()))
    return frac_pairs(R), out

print('достижимое перекрытие кандидата со существующими прошлыми')
print('ЦЕЛЬ конуса: J ~ 0.17 при |D|/N ~ 0.05-0.10')
print()
for alpha in (1.0, 0.0):
    f, rows = probe(600, 0.05, alpha)
    print(f'мир выращен alpha={alpha} (доля пар {f:.4f}), пул из 64:')
    print(f"{'затравка':>10}{'|D|/N':>9}{'J мин':>9}{'J сред':>9}{'J макс':>9}")
    for mf, sz, jmn, jmd, jmx in rows:
        print(f'{mf:>10.2f}{sz:>9.3f}{jmn:>9.4f}{jmd:>9.4f}{jmx:>9.4f}')
    print()
