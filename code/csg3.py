#!/usr/bin/env python3
"""ЧЕСТНАЯ ПОДГОНКА: x под долю связанных, а не под E|S|.

csg2.py дал при a=3: ширина совпала с засевом d=4 на 0.4 сигмы, а доля
связанных вышла 0.264 против 0.104, связи 1.254 против 1.722 (18 сигм).
Но x там подбиралось под E|S| = 2.5 при n=100 — под величину МОЕГО
выбора. Это ровно та ловушка, что ловилась сегодня не раз.

Здесь x подбирается под НАБЛЮДАЕМУЮ: доля связанных при n=1000 должна
быть 0.10, как у засева d=4. Тогда у нас два параметра (a задаёт
показатель потока, x — плотность) на две цели (доля связанных и
показатель ширины), и вопрос честный: достижимы ли обе разом.

ПРЕДРЕГИСТРАЦИЯ: если достижимы — у нас первая законная динамика с
правильной геометрией по двум независимым наблюдаемым, и надо смотреть
третью (связи). Если при r = 0.10 показатель ширины уезжает — значит
один параметр плотности не может обслужить две цели, и ограничение
семейства Райдаута–Соркина доказано измерением.

КОНТРОЛЬ: эталон (засев d=4) мерится тем же кодом в том же прогоне.
"""
import math

import numpy as np

from csg import fit
from csg2 import grow_rc, observables
from sectors import sprinkle_d


def r_at(nmax, a, lx, rs, reps=1):
    v = []
    for _ in range(reps):
        Rc = grow_rc(nmax, a, lx, rs)
        n = nmax
        v.append(int(Rc.sum()) / (n * (n - 1) / 2))
    return float(np.mean(v))


def tune_r(a, rs, target=0.104, nmax=400):
    lo, hi = -30.0, 10.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if r_at(nmax, a, mid, rs) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    print(__doc__)
    rs = np.random.default_rng(778899)
    ns = [125, 250, 500, 1000]

    ref = {n: observables(sprinkle_d(n, 4, rs)) for n in ns}
    rL, sL = fit(ns, [ref[n]['L'] for n in ns])
    rW, sW = fit(ns, [ref[n]['W'] for n in ns])
    rH, sH = fit(ns, [ref[n]['H'] for n in ns])
    print(f'ЭТАЛОН засев d=4: r(1000)={ref[1000]["r"]:.4f}, '
          f'показатели L={rL:.4f}±{sL:.3f}, W={rW:.4f}±{sW:.3f}, '
          f'H={rH:.4f}±{sH:.3f}')

    print('\nx подбирается под долю связанных 0.104 при n=400')
    print(f"{'a':>3}{'x':>12}{'r(1000)':>10}{'показ. W':>14}"
          f"{'показ. L':>14}{'показ. H':>14}")
    for a in (1, 2, 3, 4):
        lx = tune_r(a, rs)
        got = {}
        for n in ns:
            got[n] = observables(grow_rc(n, a, lx, rs))
        aW, eW = fit(ns, [got[n]['W'] for n in ns])
        aL, eL = fit(ns, [got[n]['L'] for n in ns])
        aH, eH = fit(ns, [got[n]['H'] for n in ns])
        print(f'{a:>3}{math.exp(lx):>12.4g}{got[1000]["r"]:>10.4f}'
              f'{aW:>9.4f}±{eW:<4.3f}{aL:>9.4f}±{eL:<4.3f}'
              f'{aH:>9.4f}±{eH:<4.3f}', flush=True)
        for nm, v, e, rv, re in (('W', aW, eW, rW, sW),
                                 ('L', aL, eL, rL, sL),
                                 ('H', aH, eH, rH, sH)):
            print(f'     {nm}: расхождение с эталоном '
                  f'{abs(v-rv)/math.sqrt(e**2+re**2):>5.1f}с')

    print('\nЧТЕНИЕ: если ни при каком a три показателя не сходятся с')
    print('эталоном одновременно, то один параметр плотности не может')
    print('обслужить несколько целей — и это ограничение самого')
    print('семейства, где вероятность зависит ТОЛЬКО от |S|, а не от')
    print('формы выбранного прошлого.')


if __name__ == '__main__':
    main()
