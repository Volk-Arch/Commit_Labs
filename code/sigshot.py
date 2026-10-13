#!/usr/bin/env python3
"""ДРОБОВОЙ ШУМ С ВЕСАМИ: закон sigma(S) без свободных параметров.

Форма `sigma = A sqrt(P)` с одной постоянной провалилась по константе
(дрейф 26.7%), и нуль-контроль показал, что у неё нет плеча. Разбор
дал причину: A не универсальна, она есть средневзвешенный вес

    A = k_4 sqrt( sum_m w_m^2 N_m / P ),   P = sum_m N_m,  m <= 3

У многообразных смесь N0:N1:N2:N3 даёт A ~ 12.7, у слоистых почти всё
сидит в N0 с весом -1 и выходит A ~ 0.8 — разница в 15 раз. Значит
проверяемый закон это

    sigma_шум(S) = k_4 sqrt( sum_m w_m^2 N_m )     (независимые пары)

НОЛЬ свободных параметров: все N_m замеряются, веса известны.

ПРЕДСКАЗАНИЕ до прогона. Отношение замеренного к предсказанному
обязано быть порядка единицы в ОБОИХ секторах. Если закон верен,
отношение не зависит от N и одинаково у секторов с совершенно разной
смесью весов. Отклонение отношения от единицы и есть мера корреляций
между парами.

ПОЧЕМУ У ЭТОГО ТЕСТА ЕСТЬ ПЛЕЧО, которого не было у предыдущего:
у слоистых P ~ N^2 и смесь почти чистая (один вес), у многообразных
P ~ N^{1.66} и смесь из четырёх весов с сокращением. Предсказываемая
величина различается у них в 15 раз, так что совпадение отношения
нельзя получить случайно.

КОНТРОЛЬ. Слоистый сектор считается тем же `counts_Rc`, что и
многообразный, а сам `counts_Rc` сверен с реализацией на множествах
(sigexp2.py, 12/12 и N=260 точно).
"""
import math

import numpy as np

from qcarrier import sprinkle4
from sectors import layered

K4 = 4.0 / math.sqrt(6.0)
WT = np.array([-1.0, 9.0, -16.0, 8.0])


def counts_Rc(Rc):
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    out = [n]
    for m in range(4):
        out.append(int((Rc & (Iv == m)).sum()))
    return out


def rc_sprinkle(n, rs):
    P = sprinkle4(n, rs, 1.0)
    t, X = P[:, 0], P[:, 1:]
    dt = t[None, :] - t[:, None]
    d = X[None, :, :] - X[:, None, :]
    return (dt > 0) & (dt * dt > (d * d).sum(-1))


def ens(gen, n, k, rs):
    rows = [counts_Rc(gen(n, rs)) for _ in range(k)]
    A = np.array(rows, dtype=float)
    S = K4 * (A[:, 0] + A[:, 1:5] @ WT)
    shot = K4 * np.sqrt((A[:, 1:5] * WT ** 2).sum(axis=1))
    return (float(S.std(ddof=1)), float(shot.mean()),
            float(A[:, 1:5].sum(axis=1).mean()), A[:, 1:5].mean(axis=0))


def main():
    print(__doc__)
    rs = np.random.default_rng(577215)
    sectors = (
        ('многообразные (посев 4D)', rc_sprinkle),
        ('слоистые k=4, q=0.5', lambda n, r: layered(n, 4, 0.5, r)[0]),
    )
    plan = [(200, 300), (400, 240), (800, 160), (1600, 100), (3200, 44)]
    res = {}
    for nm, gen in sectors:
        print(f'\n=== {nm} ===')
        print(f"{'N':>6}{'sigma зам.':>12}{'sigma пред.':>13}"
              f"{'отнош.':>9}{'<P>':>10}{'A=s/sqrt(P)':>13}")
        rows = []
        for n, k in plan:
            sd, sh, Pm, mix = ens(gen, n, k, rs)
            rows.append((n, sd, sh, Pm, k, mix))
            print(f'{n:>6}{sd:>12.0f}{sh:>13.0f}{sd/sh:>9.3f}'
                  f'{Pm:>10.0f}{sd/math.sqrt(Pm):>13.2f}', flush=True)
        res[nm] = rows
        r = np.array([x[1] / x[2] for x in rows])
        ks = np.array([x[4] for x in rows], float)
        er = r / np.sqrt(2 * (ks - 1))
        wm = float((r / er ** 2).sum() / (1 / er ** 2).sum())
        we = float(1 / math.sqrt((1 / er ** 2).sum()))
        Ns = np.array([x[0] for x in rows], float)
        slope = np.polyfit(np.log(Ns), np.log(r), 1)[0]
        print(f'  отношение: {wm:.3f} ± {we:.3f}, '
              f'показатель по N {slope:+.3f} '
              f'(закон требует 0)')
        pm = np.array([x[3] for x in rows])
        print(f'  <P> растёт как N^'
              f'{np.polyfit(np.log(Ns), np.log(pm), 1)[0]:.3f}, '
              f'смесь при N=3200: '
              + '/'.join(f'{v:.0f}' for v in rows[-1][5]))

    print('\n=== ПЛЕЧО: насколько тест нельзя пройти случайно ===')
    a = res['многообразные (посев 4D)'][-1]
    b = res['слоистые k=4, q=0.5'][-1]
    print(f'  предсказанная sigma при N=3200: многообразные {a[2]:.0f}, '
          f'слоистые {b[2]:.0f} — различие {a[2]/b[2]:.1f} раз')
    print(f'  A = sigma/sqrt(P):              многообразные '
          f'{a[1]/math.sqrt(a[3]):.2f}, слоистые '
          f'{b[1]/math.sqrt(b[3]):.2f} — различие '
          f'{(a[1]/math.sqrt(a[3]))/(b[1]/math.sqrt(b[3])):.1f} раз')
    print('  то есть одна постоянная A оба сектора описать не может, '
          'а закон с весами — обязан')

    print('\n=== СЛЕДСТВИЕ для показателей (это и есть ставка) ===')
    for nm in res:
        rows = res[nm]
        Ns = np.array([x[0] for x in rows], float)
        sd = np.array([x[1] for x in rows])
        ks = np.array([x[4] for x in rows], float)
        er = 1 / np.sqrt(2 * (ks - 1))
        A = np.vstack([np.log(Ns), np.ones(len(Ns))]).T
        W = np.diag(1 / er ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        be = cov @ (A.T @ W @ np.log(sd))
        print(f'  {nm:>26}: показатель sigma = {be[0]:.4f} ± '
              f'{math.sqrt(cov[0,0]):.4f}')
        for i in range(len(Ns) - 1):
            lr = math.log(Ns[i + 1] / Ns[i])
            print(f'    {int(Ns[i]):>5}->{int(Ns[i+1]):<5} местный '
                  f'{math.log(sd[i+1]/sd[i])/lr:>6.3f}')


if __name__ == '__main__':
    main()
