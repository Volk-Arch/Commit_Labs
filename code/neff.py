#!/usr/bin/env python3
"""СКОЛЬКО ИНТЕРВАЛОВ НА МИР РАБОТАЮТ НЕЗАВИСИМО.

Набор берёт по 420 интервалов из ОДНОГО сприклинга на мир. Интервалы
перекрываются, значит их шумы скоррелированы, и эффективное число может
быть сильно меньше 420. Это решает дизайн: если `n_eff << 420`, то
наращивать надо МИРЫ, а не интервалы, и запущенный набор надо
перестраивать.

Мера прямая: будь интервалы независимы, разброс средних ПО МИРАМ был бы
равен внутримировому разбросу, делённому на число интервалов:

    n_eff = Var_внутри / Var_между

Складной нож здесь не нужен — обе дисперсии считаются напрямую.
КОНТРОЛЬ: та же величина на плоском плече, где сигнала нет вовсе, —
`n_eff` обязано выйти похожим, иначе мера ловит сигнал, а не шум.

  python3 neff.py
"""
import numpy as np

from pencil import IC, IX2, IX3, IV
from pw import load


def neff(A, sd, col):
    us = np.unique(sd)
    mu, cnt, vin = [], [], []
    for u in us:
        m = sd == u
        B = A[m]
        w = B[:, IC]
        mm = (B[:, col] * w).sum() / w.sum()
        mu.append(mm); cnt.append(m.sum())
        vin.append(((B[:, col] - mm) ** 2 * w).sum() / w.sum())
    mu, cnt, vin = np.array(mu), np.array(cnt), np.array(vin)
    vbet = mu.var(ddof=1)
    vwin = vin.mean()
    n = cnt.mean()
    return n, vwin, vbet, vwin / vbet / n


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    arms = load()
    print(f"{'плечо':>14}{'интервалов/мир':>16}{'Var внутри':>13}"
          f"{'Var между':>13}{'n_eff':>9}{'доля':>8}")
    for k in sorted(arms, key=lambda x: str(x)):
        A, sd = arms[k]
        for col, cn in ((IX2, 'X2'), (IX3, 'X3')):
            n, vw, vb, frac = neff(A, sd, col)
            print(f'{f"{k[0]} {cn}":>14}{n:>16.0f}{vw:>13.4f}{vb:>13.6f}'
                  f'{frac*n:>9.0f}{frac:>8.2f}')
    print()
    print('''ЧТЕНИЕ. `доля` = n_eff / (интервалов на мир). Близко к единице —
интервалы работают независимо, и наращивать можно и то, и другое. Сильно
меньше — внутри мира они скоррелированы, и миры дороже интервалов ровно
во столько раз.''')


if __name__ == '__main__':
    main()
