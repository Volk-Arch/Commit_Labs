"""Воспроизводимость g=0.10: три засева, вдвое дольше, против мерила."""
import numpy as np
from cells3 import box, ag_of, facts, target
from cells4 import move

N, K, G, SW = 500, 5, 0.10, 120
tg = target(N, np.random.default_rng(31337))
tg2 = target(N, np.random.default_rng(999))
print(f"мерило (2 засева): доля пар {tg['f']:.4f}/{tg2['f']:.4f}  "
      f"выс {tg['hi']}/{tg2['hi']}  n0 {tg['n0']:.4f}/{tg2['n0']:.4f}  "
      f"n1 {tg['n1']:.4f}/{tg2['n1']:.4f}  n2 {tg['n2']:.4f}/{tg2['n2']:.4f}  "
      f"n3 {tg['n3']:.4f}/{tg2['n3']:.4f}  разбр {tg['cv']:.4f}/{tg2['cv']:.4f}")
print(f"\ng={G}, k={K}, {SW} проходов, три засева:")
print(f"{'засев':>7}{'доля пар':>10}{'выс.':>6}{'n0':>8}{'n1':>8}"
      f"{'n2':>8}{'n3':>8}{'разброс':>9}")
acc = []
for sd in (11, 22, 33):
    rs = np.random.default_rng(sd)
    P = box(N, K, rs); AG = ag_of(P)
    inv = np.empty((K, N), np.int32)
    for s in range(K): inv[s] = np.argsort(P[:, s])
    for w in range(SW):
        for _ in range(N):
            move(P, AG, inv, int(rs.integers(0,N)), int(rs.integers(0,K)),
                 K, G, rs)
    fc = facts(AG == K); acc.append(fc)
    print(f"{sd:>7}{fc['f']:>10.4f}{fc['hi']:>6}{fc['n0']:>8.4f}"
          f"{fc['n1']:>8.4f}{fc['n2']:>8.4f}{fc['n3']:>8.4f}{fc['cv']:>9.4f}")
m = {k: np.mean([a[k] for a in acc]) for k in ('f','hi','n0','n1','n2','n3','cv')}
print(f"{'средн':>7}{m['f']:>10.4f}{m['hi']:>6.1f}{m['n0']:>8.4f}"
      f"{m['n1']:>8.4f}{m['n2']:>8.4f}{m['n3']:>8.4f}{m['cv']:>9.4f}")
print(f"{'промах':>7}{m['f']-tg['f']:>+10.4f}{m['hi']-tg['hi']:>+6.1f}"
      f"{m['n0']-tg['n0']:>+8.4f}{m['n1']-tg['n1']:>+8.4f}"
      f"{m['n2']-tg['n2']:>+8.4f}{m['n3']-tg['n3']:>+8.4f}"
      f"{m['cv']-tg['cv']:>+9.4f}")
