#!/usr/bin/env python3
"""РАЗРЕЖЕННЫЙ СЕКТОР: решающая цифра — его σ(S).

ИЕРАРХИЯ ПО ЧИСЛУ КОНФИГУРАЦИЙ (при M = 0.25N², count = e^{M·H(q)}):

    плотные слоистые,  q = ½          e^{0.173 N²}
    разреженные,       q ~ N^{-1/2}   e^{0.06 N^{1.5} ln N}
    многообразные                     e^{d N ln N}

Плотные гасятся ТОЧНО до `|2cos(k_d/2ℏ)|^M = e^{−cN²}` в полосе
`ℏ ∈ [0.390, 0.780]` (`floor.py`). Но РАЗРЕЖЕННЫХ больше, чем
многообразных: показатель `N^{1.5}ln N` против `N ln N`.

РЕШАЕТ ОДНА ЦИФРА. Амплитуда сектора есть
`max(count·e^{−σ²/2ℏ²}, √count)`:

  * если у разреженных `σ(S)` ВЕЛИКО (`σ/ℏ ≫ 1`), работает их пол
    `√count = e^{0.03 N^{1.5} ln N}`, и они бьют многообразных
    (`e^{2N ln N}`) ОДНИМ КОЛИЧЕСТВОМ, при любом `ℏ`. Фазовый путь
    закрыт — но уже другим сектором и с более слабым разрывом, чем я
    думал утром;
  * если `σ(S)` МАЛО, их фазы когерентны, сумма по `q` имеет
    стационарную фазу, и тогда гашение возможно.

ЧТО МЕРИТСЯ: `σ(S)` разреженного семейства при фиксированном `λ` как
функция `N`, и показатель. Из `uphill.py` известно среднее
(`S/N² = +0.531` при `λ ≈ 0.45`); здесь нужен РАЗБРОС.

КОНТРОЛЬ: то же для плотного семейства, где ответ известен точно —
`σ = k_d√(M q(1−q)) ∝ N`.

  python3 sparse.py
"""
import math
import time

import numpy as np

from bd import action
from uphill import gen3

K4 = 4.0 / math.sqrt(6.0)
SP = (0.45, 0.10, 0.45)
DN = (0.25, 0.50, 0.25)


def spread(n, parts, lam_or_q, sparse, seeds, root):
    p = np.array(parts, float); p /= p.sum()
    q = (math.sqrt(lam_or_q / (p[1] * n)) if sparse else lam_or_q)
    if q > 1.0:
        return None
    v = []
    for s in range(seeds):
        A, _ = gen3(n, parts, q, np.random.default_rng(root + 7919 * s))
        v.append(action(A, 4)[0])
        del A
    v = np.array(v, float)
    sd = v.std(ddof=1)
    return v.mean(), sd, sd / math.sqrt(2.0 * (len(v) - 1)), q


def expo(NS, Y):
    x, y = np.log(np.array(NS, float)), np.log(np.array(Y, float))
    return float(np.polyfit(x, y, 1)[0])


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    NS = (600, 1200, 2400, 4800)
    print('=' * 100)
    print('КОНТРОЛЬ: плотные, q = ½. σ обязана идти как N (точно k₄√(M/4))')
    print(f"{'N':>7}{'q':>9}{'⟨S⟩/N²':>12}{'σ(S)':>14}{'σ точно':>11}"
          f"{'σ/N':>9}")
    Y = []
    for n in NS:
        r = spread(n, DN, 0.5, False, 16, 70000 + n)
        m, sd, se, q = r
        p = np.array(DN); p /= p.sum()
        sz = np.floor(p * n).astype(int); sz[-1] += n - sz.sum()
        M = sz[0] * sz[1] + sz[1] * sz[2]
        print(f'{n:>7}{q:>9.4f}{m/n**2:>12.4f}{sd:>9.1f}±{se:<4.1f}'
              f'{K4*math.sqrt(M*0.25):>11.1f}{sd/n:>9.3f}')
        Y.append(sd)
    print(f'  показатель {expo(NS, Y):.3f}  (ожидание 1.000)')
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 100)
    print('ЗАМЕР: разреженные, λ = 0.45. Вот эта цифра решает.')
    print(f"{'N':>7}{'q':>9}{'⟨S⟩/N²':>12}{'σ(S)':>14}{'σ/N':>9}"
          f"{'σ/N²':>10}")
    Y2 = []
    for n in NS:
        r = spread(n, SP, 0.45, True, 16, 71000 + n)
        if r is None:
            continue
        m, sd, se, q = r
        print(f'{n:>7}{q:>9.4f}{m/n**2:>12.4f}{sd:>9.1f}±{se:<4.1f}'
              f'{sd/n:>9.3f}{sd/n**2:>10.5f}')
        Y2.append(sd)
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    a2 = expo(NS[:len(Y2)], Y2)
    print(f'  показатель {a2:.3f}')
    print()
    print('=' * 100)
    print('СВЕДЕНИЕ: три сектора, правильные полы, ℏ в полосе')
    hb = 0.5
    print(f'  ℏ = {hb} (середина первой полосы), '
          f'|2cos(k₄/2ℏ)| = {abs(2*math.cos(K4/(2*hb))):.4f}')
    print()
    print(f"{'N':>8}{'ln|Z| плотные':>17}{'ln|Z| разрежен.':>18}"
          f"{'ln|Z| многообр.':>18}{'кто больше':>16}")
    for n in (1e3, 1e6, 1e12, 1e60):
        M = 0.25 * n ** 2
        lz_d = M * math.log(abs(2 * math.cos(K4 / (2 * hb))))
        lz_s = 0.5 * 0.0625 * n ** 1.5 * math.log(n)
        lz_m = 0.5 * 4.0 * n * math.log(n)
        who = max((lz_d, 'плотные'), (lz_s, 'разрежен.'),
                  (lz_m, 'многообр.'))[1]
        print(f'{n:>8.0e}{lz_d:>17.3e}{lz_s:>18.3e}{lz_m:>18.3e}{who:>16}')
    print()
    print('''  (ln|Z| плотных ОТРИЦАТЕЛЕН в полосе — точное гашение;
   у двух других взят пол √count, то есть ½·ln count. Разреженные
   побеждают многообразных, если показатель σ у них велик — что и
   проверялось выше.)''')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
