"""Воспроизводимость k=6: три засева правила против трёх засевов мерила."""
import numpy as np
from cells3 import box, ag_of, facts, target
from cells5 import step

def prof(R):
    N = len(R); h = np.ones(N, np.int64)
    for i in np.argsort(R.sum(0), kind='stable'):
        i = int(i); pre = np.flatnonzero(R[:, i])
        if len(pre): h[i] = h[pre].max() + 1
    b = np.bincount(h)[1:]
    return np.array([float(x)/N for x in b])

def fmt(p, m=7):
    return ' '.join(f'{p[i]:.3f}' if i < len(p) else '  .  ' for i in range(m))

N, K, SW = 400, 6, 400
print(f'N={N}, k={K}, {SW} проходов, НУЛЬ ручек')
print(f"\n{'источник':>14}{'доля пар':>10}{'n0':>8}{'n1':>8}{'выс.':>6}"
      f"{'разброс':>9}   профиль слоёв")
T = []
for sd in (31337, 999, 55555):
    t = target(N, np.random.default_rng(sd)); T.append(t)
    rs = np.random.default_rng(sd)
    out, got = [], 0
    while got < N:
        m = (N-got)*9+128; Y = np.empty((m,4))
        Y[:,0] = rs.uniform(0,1,m); Y[:,1:] = rs.uniform(-.5,.5,(m,3))
        rad = np.linalg.norm(Y[:,1:],axis=1)
        Y = Y[(Y[:,0]>=rad)&(1-Y[:,0]>=rad)]
        if len(Y): out.append(Y); got += len(Y)
    X = np.vstack(out)[:N]; X = X[np.argsort(X[:,0])]
    dt = X[:,0][None,:]-X[:,0][:,None]
    dx = np.linalg.norm(X[:,None,1:]-X[None,:,1:],axis=2)
    print(f"{'сприклинг':>14}{t['f']:>10.4f}{t['n0']:>8.4f}{t['n1']:>8.4f}"
          f"{t['hi']:>6}{t['cv']:>9.4f}   {fmt(prof((dt>0)&(dt>dx)))}")
A = []
for sd in (101, 202, 303):
    rs = np.random.default_rng(sd)
    P = box(N, K, rs); AG = ag_of(P)
    inv = np.empty((K, N), np.int32)
    for s in range(K): inv[s] = np.argsort(P[:, s])
    for _ in range(SW):
        for _ in range(N): step(P, AG, inv, K, rs)
    R = AG == K; fc = facts(R); A.append(fc)
    print(f"{'правило':>14}{fc['f']:>10.4f}{fc['n0']:>8.4f}{fc['n1']:>8.4f}"
          f"{fc['hi']:>6}{fc['cv']:>9.4f}   {fmt(prof(R))}")
g = lambda ks: (np.mean([a[ks] for a in A]), np.mean([t[ks] for t in T]))
print()
for ks in ('f','n0','n1','hi','cv'):
    a, t = g(ks)
    print(f'  {ks:>3}: правило {a:.4f}   мерило {t:.4f}   '
          f'промах {100*(a-t)/max(abs(t),1e-9):+.1f} %')
