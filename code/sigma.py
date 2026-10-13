#!/usr/bin/env python3
"""ПРЕДУСЛОВИЕ ФАЗОВОГО ОТБОРА: показатель σ(S) по N.

ЗАЧЕМ ЭТО ПЕРВЫМ. Фазовый отбор работает, если слоистые порядки гасятся
СИЛЬНЕЕ многообразных. Для слоистых подавление считается ТОЧНО: при
`S = k_d(N − N₀)` сумма по всем конфигурациям связей раскладывается,

    Z = e^{i k_d N/ℏ} (1 + e^{−i k_d/ℏ})^M,   |Z| = |2cos(k_d/2ℏ)|^M,

против числа конфигураций `2^M`, где `M = a₂(a₁+a₃)N²`. Это равносильно
гауссовой оценке `e^{−σ²/2ℏ²}` с `σ(S) = k_d√(M q(1−q)) ∝ N` — то есть
подавление идёт как `e^{−O(N²)/ℏ²}`.

У МНОГООБРАЗНЫХ порядков действие тоже сильно флуктуирует, и по
вчерашним трём точкам (`bd.py`: 843, 1699, 5033 при N = 400, 800, 1600)
показатель выходит около 1.29. Если он БОЛЬШЕ единицы, многообразные
гасятся как `e^{−N^{2a}/ℏ²}` с `2a > 2`, то есть СИЛЬНЕЕ слоистых, и
фазовый отбор работает против нас. Весь расчёт висит на этой цифре.

    a = 1   →  состязание постоянных, `ℏ` решает, полоса существует
    a > 1   →  фазовый отбор закрыт, и закрыт окончательно

КОНТРОЛЬ, у которого ответ известен заранее: у слоистого порядка при
постоянном `q` величина `N₀` биномиальна, поэтому
`σ(S) = k_d√(M q(1−q))` ТОЧНО, то есть показатель обязан выйти 1.000.
Если прибор его не воспроизводит, мерить многообразные нельзя.

Ошибка показателя — складным ножом по точкам `N`.

  python3 sigma.py
"""
import math
import time

import numpy as np

from bd import action
from sprinkle import sprinkle_d
from uphill import gen3

K4 = 4.0 / math.sqrt(6.0)


def spread(gen, n, dim, seeds, root, **kw):
    v = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 7919 * s), **kw)
        if isinstance(A, tuple):
            A = A[0]
        v.append(action(A, dim)[0])
        del A
    v = np.array(v, float)
    sd = v.std(ddof=1)
    return v.mean(), sd, sd / math.sqrt(2.0 * (len(v) - 1))


def expo(NS, Y, E):
    x = np.log(np.asarray(NS, float))
    y = np.log(np.asarray(Y, float))
    e = np.asarray(E, float) / np.asarray(Y, float)
    w = 1.0 / np.maximum(e, 1e-12) ** 2
    M = np.vstack([np.ones_like(x), x]).T

    def fit(k):
        Mk, wk, yk = M[k], w[k], y[k]
        C = np.linalg.inv(Mk.T @ (wk[:, None] * Mk))
        return C @ (Mk.T @ (wk * yk))

    full = fit(np.ones(len(x), bool))
    jk = np.array([fit(np.arange(len(x)) != i)[1] for i in range(len(x))])
    se = math.sqrt((len(x) - 1) / len(x) * ((jk - jk.mean()) ** 2).sum())
    return float(full[1]), se, math.exp(float(full[0]))


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    PARTS = (0.25, 0.5, 0.25)
    p = np.array(PARTS, float)
    print('=' * 100)
    print('КОНТРОЛЬ: слоистый порядок, q = 0.5. Показатель ОБЯЗАН быть 1.000')
    print(f"{'N':>7}{'⟨S⟩':>14}{'σ(S) замер':>18}{'σ точно':>12}{'отн.':>9}")
    NS1 = (400, 800, 1600, 3200)
    Y, E = [], []
    for n in NS1:
        sz = np.floor(p * n).astype(int); sz[-1] += n - sz.sum()
        M = sz[0] * sz[1] + sz[1] * sz[2]
        exact = K4 * math.sqrt(M * 0.25)
        m, sd, se = spread(gen3, n, 4, 24, 1000 + n, parts=PARTS, q=0.5)
        Y.append(sd); E.append(se)
        print(f'{n:>7}{m:>14.0f}{sd:>12.1f}±{se:<5.1f}{exact:>12.1f}'
              f'{sd/exact:>9.3f}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    a, ae, c = expo(NS1, Y, E)
    print(f'  ПОКАЗАТЕЛЬ = {a:.3f} ± {ae:.3f}   (ожидание 1.000, '
          f'{(a-1)/max(ae,1e-9):+.1f} σ)')
    print()
    print('=' * 100)
    print('ЗАМЕР: многообразные порядки (сприклинг 4D). Вот эта цифра решает.')
    print(f"{'N':>7}{'⟨S⟩':>14}{'σ(S)':>18}{'σ/N':>10}{'σ/N^1.3':>11}")
    NS2 = (400, 800, 1600, 3200, 6400)
    SD = {400: 48, 800: 48, 1600: 32, 3200: 20, 6400: 8}
    Y2, E2 = [], []
    for n in NS2:
        m, sd, se = spread(sprinkle_d, n, 4, SD[n], 2000 + n, d=4)
        Y2.append(sd); E2.append(se)
        print(f'{n:>7}{m:>14.0f}{sd:>12.1f}±{se:<5.1f}{sd/n:>10.3f}'
              f'{sd/n**1.3:>11.4f}')
        print(f'    ({time.time()-t0:.0f} с, {SD[n]} засевов)', flush=True)
    a2, ae2, c2 = expo(NS2, Y2, E2)
    print(f'  ПОКАЗАТЕЛЬ = {a2:.3f} ± {ae2:.3f}')
    print()
    print('=' * 100)
    print('ЧТО ЭТО ЗНАЧИТ')
    hard = (a2 - 1.0) / max(ae2, 1e-9)
    print(f'''
   слоистые:     σ ∝ N^{a:.3f} ± {ae:.3f}  (точно 1, контроль пройден)
   многообразные: σ ∝ N^{a2:.3f} ± {ae2:.3f}   ({hard:+.1f} σ от единицы)
''')
    if a2 - 1.0 > 3 * ae2:
        print('   ПОКАЗАТЕЛЬ БОЛЬШЕ ЕДИНИЦЫ. Многообразные гасятся как')
        print(f'   e^{{−N^{2*a2:.2f}/ℏ²}}, слоистые лишь как e^{{−N²/ℏ²}}.')
        print('   Фазовый отбор работает ПРОТИВ многообразных, и закрыт.')
    elif abs(a2 - 1.0) <= 3 * ae2:
        print('   ПОКАЗАТЕЛЬ СОВМЕСТИМ С ЕДИНИЦЕЙ. Оба сектора гасятся как')
        print('   e^{−O(N²)/ℏ²}, значит решают ПОСТОЯННЫЕ, и полоса по `ℏ`')
        print('   может существовать. Считать её вторым прогоном.')
    else:
        print('   ПОКАЗАТЕЛЬ МЕНЬШЕ ЕДИНИЦЫ — многообразные гасятся слабее,')
        print('   и это лучший из возможных исходов.')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
