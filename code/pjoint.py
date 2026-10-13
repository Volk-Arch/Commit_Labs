#!/usr/bin/env python3
"""СОВМЕСТНАЯ ПОДГОНКА КАРАНДАША: своя кривизна каждому, помеха общая.

ПОЧЕМУ ТАК, А НЕ ИНАЧЕ — две прежние попытки и что в них было не так.

  ПОПЫТКА 1: подгонять `alpha, beta` вместе с мешающими членами, свалив
  плоское и кривое в одну выборку. Вышло вырождение (число
  обусловленности 6e8) и «отношение» ровно `-1.00` в обоих семействах.
  Причина ДВОЙНАЯ. Во-первых, `g_1 - g_2 = sqrt(C_1)` тождественно, а над
  нашим диапазоном `C_1` степени `sqrt(C_1)`, `1`, `1/sqrt(C_1)` почти
  коллинеарны. Во-вторых — и это грубее — я дал ОДНИ `alpha, beta` на все
  строки, включая плоские, где они равны нулю по построению: плоские
  строки просто тянули сигнал к нулю.

  ПОПЫТКА 2: вычитать плоское поячеечно. Вырождение ушло
  (обусловленность 72), но средние ячеек плоского сами шумны
  (±0.002...0.004) и сравнимы с сигналом в полосах МАЛОГО наклона — а
  именно они определяют `alpha`. У линейного семейства ошибка съела всё.

  ЗДЕСЬ: помеха физически ГЛАДКАЯ (её источники — неточность `c_2` и
  `c_3`, отбор по вложению в ящик, конечность `C_1`), поэтому она
  описывается четырьмя параметрами

      N(v², C_1) = n_0 + n_1/sqrt(C_1) + n_2 v² + n_3 v²/sqrt(C_1)

  по всему плоскому плечу сразу, а не 24 независимыми средними. Это
  режет её шум в sqrt(24/4) = 2.4 раза. Кривизна же получает СВОИ
  `alpha, beta` в каждом семействе:

      X_2 = alpha_f g_1 [строка семейства f] + beta_f g_2 [строка f]
            + N(v², C_1),

      g_1 = sqrt(C_1)/(1-v²),    g_2 = sqrt(C_1) v²/(1-v²)

  Плоские строки имеют нули в первых столбцах и потому закрепляют ТОЛЬКО
  помеху — ради этого они и собирались.

ЧТО ИЗ ЭТОГО ДОСТАЁТСЯ. `alpha_f = v_2 k R_etaeta l²`,
`beta_f = v_2 k R_xx l²`, значит отношение `R_xx/R_etaeta` не требует
калибровки вовсе, а скаляр берётся свёрткой:

    R = ( -R_etaeta + 3 R_xx ) / a²

У линейного семейства он обязан выйти НУЛЁМ — точным сокращением двух
НЕНУЛЕВЫХ измеренных чисел (`R_etaeta = 3h²`, `R_xx = h²`), — а у
квадратичного `6c`. Это проверяет не слепоту замера, а его верность.

КОНТРОЛЬ КОНВЕЙЕРА: половина плоского объявляется «семейством», и её
`alpha, beta` обязаны выйти нулями. Проверка всей цепочки — столбцов,
весов, складного ножа — на данных, где кривизны нет по построению.

  python3 pjoint.py
"""
import math
import pickle

import numpy as np

from pencil import SCRATCH, KFAC, V2CAL, V3CAL, truth, IC, IX2, IX3, IV


def plan(arms):
    """arms: список (A, sd, кривое ли). Возвращает A, D, w, sd."""
    A = np.vstack([a for a, _, _ in arms])
    sd = np.concatenate([s + 100000 * i
                         for i, (_, s, _) in enumerate(arms)])
    s = np.sqrt(A[:, IC])
    g = 1.0 / (1.0 - A[:, IV])
    g1, g2 = s * g, s * A[:, IV] * g
    cols, off = [], 0
    for a, _, curved in arms:
        if curved:
            m = np.zeros(len(A))
            m[off:off + len(a)] = 1.0
            cols += [g1 * m, g2 * m]
        off += len(a)
    cols += [np.ones(len(A)), 1.0 / s, A[:, IV], A[:, IV] / s]
    return A, np.column_stack(cols), A[:, IC], sd


def solve(D, y, w, mask=None):
    if mask is not None:
        D, y, w = D[mask], y[mask], w[mask]
    Dw = D * w[:, None]
    return np.linalg.solve(D.T @ Dw, Dw.T @ y)


def do(arms, names, truths, col, vcal):
    A, D, w, sd = plan(arms)
    y = A[:, col]
    full = solve(D, y, w)
    us = np.unique(sd)
    jk = np.array([solve(D, y, w, sd != u) for u in us])
    k = len(us)
    M = D.T @ (D * w[:, None])
    print(f'    обусловленность плана {np.linalg.cond(M):.1f}, '
          f'параметров {D.shape[1]}, строк {len(A)}')

    def se(x):
        return math.sqrt((k - 1) / k * ((x - x.mean()) ** 2).sum())

    f = vcal * KFAC
    for i, (name, tr) in enumerate(zip(names, truths)):
        al, be = full[2 * i], full[2 * i + 1]
        jal, jbe = jk[:, 2 * i], jk[:, 2 * i + 1]
        Ree_t, Rxx_t, R_t = tr
        rt = Rxx_t / Ree_t if Ree_t else float('nan')
        r0, er = be / al, se(jbe / jal)
        s0, es = (-al + 3 * be) / f, se((-jal + 3 * jbe) / f)
        print(f'    {name}:')
        print(f'      R_etaeta l²   = {al/f:+.5f} +- {se(jal)/abs(f):.5f}'
              f'   (истина {Ree_t:+.5f})')
        print(f'      R_xx l²       = {be/f:+.5f} +- {se(jbe)/abs(f):.5f}'
              f'   (истина {Rxx_t:+.5f})')
        print(f'      R_xx/R_etaeta = {r0:+.3f} +- {er:.3f}'
              f'   (истина {rt:+.3f})   {abs(r0-rt)/max(er,1e-12):.1f} сигм')
        print(f'      R l² свёрткой = {s0:+.5f} +- {es:.5f}'
              f'   (истина {R_t:+.5f})   {abs(s0-R_t)/max(es,1e-12):.1f} сигм'
              f'   [от нуля {abs(s0)/max(es,1e-12):.1f} сигм]')


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/pencil.pkl', 'rb') as fh:
        res = pickle.load(fh)
    fk = [k for k in res if k[0] == 'flat'][0]
    curved = [k for k in res if k[0] != 'flat']
    print('набор:', {f'{k[0]} {k[1]}': len(res[k][0]) for k in res})
    Af, sdf = res[fk]
    us = np.unique(sdf)
    h1 = np.isin(sdf, us[::2])

    for col, cname, vcal in ((IX2, 'X_2', V2CAL), (IX3, 'X_3', V3CAL)):
        print()
        print('=' * 74)
        print(f'{cname}')
        print('  КОНТРОЛЬ КОНВЕЙЕРА: половина плоского объявлена семейством')
        do([(Af[h1], sdf[h1], True), (Af[~h1], sdf[~h1], False)],
           ['половина плоского'], [(1.0, 0.0, 0.0)], col, vcal)
        print('  СЕМЕЙСТВА СОВМЕСТНО, помеха общая')
        do([(res[k][0], res[k][1], True) for k in curved]
           + [(Af, sdf, False)],
           [f'{k[0]} {k[1]}' for k in curved],
           [truth(*k) for k in curved], col, vcal)


if __name__ == '__main__':
    main()
