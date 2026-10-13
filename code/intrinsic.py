"""Внутренняя ли величина? Та же масса, разные интервалы.

На оси количества при M=160 размерность дала 0.0701, на оси сжатия —
0.0137. Масса одна и та же, отличались интервалы (V_bg 735 против 1682).

Если наблюдаемая зависит от интервала, в котором её мерят, она не
свойство массы, а свойство пары «масса плюс измеритель». Тогда стена
не в том, что наблюдаемые спорят, а в том, что ни одна не внутренняя.

Проверка прямая: одна и та же масса, пары разбиты по объёму интервала.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from masstest import observables
from linkfield import ball, pick_pairs, CENTER
from core import rel_of

M, R_BALL, K = 160, 0.07, 4
N = 30000
REALS = 2


def collect(n_pairs=60, vol_cap=2600, seed=5):
    rs = np.random.default_rng(seed)
    Xb = rs.random((N, K)).astype(np.float32)
    rb = np.random.default_rng(seed + 77)
    masses = [ball(M, K, R_BALL, rb) for _ in range(REALS)]
    # широкий пул, чтобы объёмы интервалов разошлись
    ia = np.flatnonzero((Xb < CENTER - R_BALL).all(1))
    ib = np.flatnonzero((Xb > CENTER + R_BALL).all(1))
    ia = ia[np.argsort(-Xb[ia].max(1))][:400]
    ib = ib[np.argsort(Xb[ib].min(1))][:400]
    pr = list(zip(rs.choice(ia, n_pairs), rs.choice(ib, n_pairs)))
    out = []
    for a, b in pr:
        xa, xb = Xb[a], Xb[b]
        m = (Xb > xa).all(1) & (Xb < xb).all(1)
        m[a] = m[b] = False
        iv = np.flatnonzero(m)
        if len(iv) > vol_cap or len(iv) < 150:
            continue
        Xbg, nb = Xb[iv], len(iv)
        b0 = observables(np.vstack([Xbg, xa[None], xb[None]]), nb, 0)
        for B in masses:
            o = observables(np.vstack([Xbg, B, xa[None], xb[None]]), nb, M)
            out.append((nb,
                        (b0['links_bg'] - o['links_bg']) / b0['links_bg'],
                        o['tau'] - b0['tau'],
                        o['dim'] - b0['dim'],
                        o['h_add']))
    return np.array(out, float)


A = collect()
print(f'масса M={M}, радиус {R_BALL}, k={K}, N={N}, измерений {len(A)}')
print(f'высота массы {A[:, 4].mean():.2f} ± {A[:, 4].std():.2f} '
      f'(одна и та же масса во всех парах)')
print()
V = A[:, 0]
edges = np.quantile(V, [0, 0.25, 0.5, 0.75, 1.0])
print(f"{'V_bg':>14}{'M/V_bg':>9}{'порвано связей':>18}"
      f"{'задержка Δτ':>16}{'размерность':>18}{'измер.':>8}")
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (V >= lo) & (V <= hi)
    if m.sum() < 4:
        continue
    f = lambda c: (A[m, c].mean(), A[m, c].std() / np.sqrt(m.sum()))
    bk, tt, dd = f(1), f(2), f(3)
    print(f'{f"{lo:.0f}-{hi:.0f}":>14}{M/A[m,0].mean():>9.3f}'
          f'{f"{bk[0]:.5f} ± {bk[1]:.5f}":>18}'
          f'{f"{tt[0]:.2f} ± {tt[1]:.2f}":>16}'
          f'{f"{dd[0]:.5f} ± {dd[1]:.5f}":>18}{int(m.sum()):>8}')
    rows.append((A[m, 0].mean(), bk[0], tt[0], dd[0]))

print()
print('ЗАВИСИМОСТЬ ОТ ИЗМЕРИТЕЛЯ: показатель по объёму интервала')
print('  внутренняя величина дала бы показатель 0')
if len(rows) >= 3:
    lv = np.log([r[0] for r in rows])
    for idx, nm in ((1, 'порвано связей'), (2, 'задержка Δτ'),
                    (3, 'размерность')):
        y = np.array([r[idx] for r in rows], float)
        if (y > 0).all():
            p = np.polyfit(lv, np.log(y), 1)[0]
            print(f'{nm:>18}: {p:+.2f}')
        else:
            print(f'{nm:>18}: есть неположительные, показатель не считаю')
