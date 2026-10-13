#!/usr/bin/env python3
"""УСТОЙЧИВОСТЬ ОЦЕНЩИКА sigma(S) — проверка перед любым показателем.

Повод. При N=1600 два прогона одним кодом дали sigma(S) = 4228 и 2928.
При 90 засевах ожидаемая ошибка sigma равна 1/sqrt(2(k-1)) = 7.5%,
а расхождение 44% — то есть в шесть раз больше. Если это правда,
то все три числа в споре (архивное 0.799, вчерашнее 0.967, сегодняшнее
0.895) суть показания неустойчивого оценщика, и спорить не о чем.

ПРЕДСКАЗАНИЕ, записанное до прогона: если распределение S близко к
нормальному, разброс sigma по независимым партиям равен 1/sqrt(2(k-1)),
и наблюдаемый разброс не превысит его больше чем вдвое. Если хвосты
тяжёлые, наблюдаемый разброс будет много больше, а избыточный эксцесс
окажется большим и положительным.

Три диагностики:
  1. sigma по 10 независимым партиям при фиксированном N — наблюдаемый
     разброс против ожидаемого;
  2. избыточный эксцесс объединённой выборки (для нормального 0);
  3. устойчивая мера ширины — межквартильный размах, нормированный на
     1.349 (для нормального равен sigma) — она к хвостам нечувствительна.
     Если показатель по IQR чистый, а по sigma скачет, мерить надо IQR.
"""
import math

import numpy as np

from qcarrier import sprinkle4
from sigexp import counts4, K4


def sample(n, k, rs):
    rows = [counts4(sprinkle4(n, rs, 1.0)) for _ in range(k)]
    A = np.array(rows, dtype=float)
    return K4 * (A[:, 0] - A[:, 1] + 9 * A[:, 2] - 16 * A[:, 3]
                 + 8 * A[:, 4])


def iqr_sigma(x):
    q1, q3 = np.percentile(x, [25, 75])
    return (q3 - q1) / 1.3490


def main():
    print(__doc__)
    rs = np.random.default_rng(31337)

    print('ДИАГНОСТИКА 1-2: партии при фиксированном N')
    for n, nb, k in ((400, 10, 60), (1600, 10, 60)):
        sg, iq, pool = [], [], []
        for _ in range(nb):
            S = sample(n, k, rs)
            sg.append(float(S.std(ddof=1)))
            iq.append(float(iqr_sigma(S)))
            pool.append(S)
        sg = np.array(sg)
        iq = np.array(iq)
        P = np.concatenate(pool)
        z = (P - P.mean()) / P.std(ddof=1)
        kurt = float((z ** 4).mean() - 3.0)
        skew = float((z ** 3).mean())
        exp_rel = 1.0 / math.sqrt(2 * (k - 1))
        print(f'\n  N={n}, {nb} партий по {k} засевов')
        print('    sigma по партиям: '
              + ', '.join(f'{v:.0f}' for v in sg))
        print(f'    разброс sigma = {sg.std(ddof=1)/sg.mean()*100:.1f}% '
              f'| ожидаемый {exp_rel*100:.1f}% '
              f'| отношение {sg.std(ddof=1)/sg.mean()/exp_rel:.2f}')
        print(f'    разброс IQR   = {iq.std(ddof=1)/iq.mean()*100:.1f}% '
              f'| ожидаемый {1.17*exp_rel*100:.1f}% '
              f'| отношение {iq.std(ddof=1)/iq.mean()/(1.17*exp_rel):.2f}')
        print(f'    объединённая выборка {len(P)}: избыточный эксцесс '
              f'{kurt:+.2f} (норм. 0), асимметрия {skew:+.2f}')
        print(f'    макс |z| = {np.abs(z).max():.1f} '
              f'(у нормального при {len(P)} точках ждём ~3.5)')
        print(f'    sigma/IQR-сигма = {sg.mean()/iq.mean():.2f} '
              f'(у нормального 1.00)')

    print('\nДИАГНОСТИКА 3: показатель по устойчивой мере против sigma')
    plan = [(200, 300), (400, 240), (800, 160), (1600, 100), (3200, 48)]
    Ns, sd, iq, ks = [], [], [], []
    for n, k in plan:
        S = sample(n, k, rs)
        Ns.append(n)
        sd.append(float(S.std(ddof=1)))
        iq.append(float(iqr_sigma(S)))
        ks.append(k)
        print(f'  N={n:>5} k={k:>4}  sigma={sd[-1]:>8.0f}  '
              f'IQR-сигма={iq[-1]:>8.0f}', flush=True)
    lx = np.log(Ns)
    for nm, y, fac in (('sigma', sd, 1.0), ('IQR-сигма', iq, 1.17)):
        y = np.array(y)
        err = fac / np.sqrt(2 * (np.array(ks) - 1))
        A = np.vstack([lx, np.ones(len(lx))]).T
        W = np.diag(1.0 / err ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        be = cov @ (A.T @ W @ np.log(y))
        se = math.sqrt(cov[0, 0])
        rr = (np.log(y) - A @ be) / err
        chi = float((rr ** 2).sum() / (len(lx) - 2))
        print(f'  {nm:>10}: показатель {be[0]:.4f} ± {se:.4f}, '
              f'хи2/d.o.f. = {chi:.1f}')
        print(f'  {"":>10}  остатки: '
              + ', '.join(f'{v:+.1f}' for v in rr))


if __name__ == '__main__':
    main()
