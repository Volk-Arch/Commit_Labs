#!/usr/bin/env python3
"""РАВНОВЕСИЕ ФАЗЫ: не артефакт ли несовпадение.

critbeta.py: в точке перехода (пик теплоёмкости на beta=0.04 при N=24)
хи2 фазы против семейства засевов = 113.4 при поле контроля 0.1-3.4, и
хи2 растёт с N как 8.9, 30.0, 46.2, 111.2. При beta=0 хи2=3.8, то есть
БЕЗ действия ансамбль геометричен, а с действием нет.

Прежде чем это записывать, последний подвох: равновесие. Доля принятых
на 0.04 равна 0.234 - приемлемо, но если цепь не сошлась, наблюдаемые
сдвинуты и хи2 раздут. Проверка стандартная: удвоенный прогон и другое
зерно обязаны дать те же значения.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ: совпадение с прежним (r=0.465±0.005,
H=4.09±0.06, хи2=113.4) => равновесие есть, вывод стоит. Сдвиг =>
цепь не сошлась, и число 111 ничего не значит.
"""
import numpy as np

from critphase import BET, summarize
from critbeta import ladder
from critgeom3 import model, stats, best, box, KEYS
from sectors import sprinkle_d

DS = (2, 3, 4)


def main():
    print(__doc__)
    n = 24
    rs = np.random.default_rng(878889)
    mod = model(n, 120, rs)
    dias = {d: stats(lambda: sprinkle_d(n, d, rs), 120, rs) for d in DS}
    m, e = stats(lambda: box(n, 2, 2.0, rs), 120, rs)
    c0, g0, _ = best(m, e, mod, dias)
    print(f'КОНТРОЛЬ d=2 a=2.0 -> {g0}, хи2={c0:.1f} '
          f'=> {"ОК" if c0 < 8 else "СБОЙ"}')

    k = BET.index(0.04)
    print(f'\nпрежде: beta=0.04, 900 развёрток, зерно 606162')
    print(f'        r=0.465±0.005  H=4.09±0.06  хи2=113.4')
    recs, acts, rates = ladder(n, 1800, rs)
    tm, te = {}, {}
    for key in KEYS:
        mu, er = summarize(recs[k], key)
        tm[key], te[key] = float(mu), float(er)
    chi, geo, sig = best(tm, te, mod, dias)
    print(f'теперь: beta=0.04, 1800 развёрток, зерно 878889, '
          f'принято {rates[k]:.3f}')
    print('        ' + '  '.join(f'{key}={tm[key]:.3f}±{te[key]:.3f}'
                                 for key in KEYS))
    print(f'        хи2={chi:.1f}, геометрия {geo}')
    print('        отклонения: '
          + '  '.join(f'{key}={sig[key]:+.1f}σ' for key in KEYS))


if __name__ == '__main__':
    main()
