#!/usr/bin/env python3
"""ПОХОЖ ЛИ РОСТ ПРИ t_k = x^k/(k!)^3 НА ЧЕТЫРЁХМЕРИЕ — четыре наблюдаемых.

csg.py показал, что фиксированные связи Райдаута–Соркина дают нужный
поток: t_k = x^k/(k!)^{d-1} даёт E|S| ~ n^{1/d} без подгонки p под N.
Но ширина — одна наблюдаемая, и совпадение по ней ещё не четырёхмерие.

ПРЕДРЕГИСТРАЦИЯ (записано до прогона). Если a=3 правда подражает d=4:
  * доля связанных r -> 0.1 (Майрхейм–Майер для d=4);
  * высота ~ n^{1/4};
  * ширина ~ n^{3/4};
  * связи ~ n^{3/2} (то же, что даёт двухточечный интеграл, `nm-doc.md`).
И в общем виде при a = d-1 должно быть r -> значение Майрхейма–Майера
для этого d, высота ~ n^{1/d}, ширина ~ n^{1-1/d}.

ЭТАЛОН: засев d=4 на том же размахе, теми же счётчиками. Это главное —
сравниваются не числа с теорией, а два ансамбля одним кодом.

КОНТРОЛЬ: a=0 обязан дать перколяцию с постоянным p (проверено точно в
csg.py: E|S| = n·p до третьего знака).
"""
import math

import numpy as np

from csg import grow, tune_x, fit
from sectors import sprinkle_d
from width3 import width


def observables(Rc):
    n = Rc.shape[0]
    R = int(Rc.sum())
    F = Rc.astype(np.float32)
    two = (F @ F) > 0.5
    L = int((Rc & ~two).sum())
    hgt = np.zeros(n, np.int64)
    order = np.argsort(Rc.sum(1))[::-1]
    for _ in range(2):
        for x in order:
            pre = np.nonzero(Rc[:, x])[0]
            hgt[x] = (hgt[pre].max() + 1) if len(pre) else 0
    return dict(r=R / (n * (n - 1) / 2), L=L, H=int(hgt.max() + 1),
                W=width(Rc))


def grow_rc(nmax, a, lx, rs):
    Rc = np.zeros((nmax, nmax), bool)
    from scipy.special import logsumexp
    from csg import logw
    for n in range(1, nmax):
        lw = logw(n, a, lx)
        p = np.exp(lw - logsumexp(lw))
        k = int(rs.choice(n + 1, p=p))
        if k:
            S = rs.choice(n, size=k, replace=False)
            sel = np.zeros(n, bool)
            sel[S] = True
            Rc[:n, n] = sel | Rc[:n, :n][:, sel].any(axis=1)
    return Rc


def main():
    print(__doc__)
    rs = np.random.default_rng(55544433)
    ns = [125, 250, 500, 1000]

    print('ЭТАЛОН: засев d=4, те же счётчики')
    ref = {}
    for n in ns:
        o = observables(sprinkle_d(n, 4, rs))
        ref[n] = o
        print(f'  n={n:>5}: r={o["r"]:.4f}  связей={o["L"]:>6}  '
              f'высота={o["H"]:>3}  ширина={o["W"]:>4}')
    for key, pred in (('L', 1.5), ('H', 0.25), ('W', 0.75)):
        a_, se = fit(ns, [ref[n][key] for n in ns])
        print(f'  показатель {key}: {a_:.4f} ± {se:.3f} '
              f'(теория {pred})')
    print(f'  доля связанных на конце: {ref[1000]["r"]:.4f} '
          f'(Майрхейм–Майер d=4: 0.1)')

    print('\nРОСТ при t_k = x^k/(k!)^a')
    for a in (0, 2, 3):
        lx = tune_x(a)
        print(f'\n  a = {a}:')
        got = {}
        for n in ns:
            Rc = grow_rc(n, a, lx, rs)
            got[n] = observables(Rc)
            o = got[n]
            print(f'    n={n:>5}: r={o["r"]:.4f}  связей={o["L"]:>6}  '
                  f'высота={o["H"]:>3}  ширина={o["W"]:>4}', flush=True)
        for key, pred in (('L', 1.5), ('H', 0.25), ('W', 0.75)):
            a_, se = fit(ns, [got[n][key] for n in ns])
            ra, rse = fit(ns, [ref[n][key] for n in ns])
            print(f'    показатель {key}: {a_:.4f} ± {se:.3f}  '
                  f'| эталон {ra:.4f} ± {rse:.3f}  '
                  f'| расхождение '
                  f'{abs(a_-ra)/math.sqrt(se**2+rse**2):.1f}с')
        print(f'    доля связанных на конце: {got[1000]["r"]:.4f} '
              f'| эталон {ref[1000]["r"]:.4f}')


if __name__ == '__main__':
    main()
