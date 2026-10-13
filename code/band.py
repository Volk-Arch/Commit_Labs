#!/usr/bin/env python3
"""ПЕРЕВОРАЧИВАЕТ ЛИ СГЛАЖИВАНИЕ НЕРАВЕНСТВО? Один двоичный вопрос.

ОТКУДА. `sigma.py` дал: у обоих секторов `σ(S) ∝ N` (показатели
0.903 ± 0.065 у слоистых при точном ответе 1, и 0.995 ± 0.043 у
сприклинга). Значит оба гасятся как `e^{−O(N²)/ℏ²}`, и решают
ПОСТОЯННЫЕ. Условие выписывается точно:

  многообразные побеждают ⟺ count_m·e^{−σ_m²/2ℏ²} > count_л·e^{−σ_л²/2ℏ²}
                          ⟺ σ_л² − σ_м² > 2ℏ²·ln(count_л/count_м)
                          ⟺ (s_л² − s_м²) > 0.346·ℏ²       [σ = s·N]

где `ln count_л = (N²/4)ln2 = 0.173N²` (Клейтман–Ротшильд), а
`ln count_м = O(N ln N)` (граница Милнора–Тома) — на `N²` это нуль.

ТРЕБУЕТСЯ `s_л > s_м`, то есть слоистые должны шуметь СИЛЬНЕЕ. У резкого
действия наоборот: `s_л = k₄√(M/4)/N = 0.408`, `s_м = 2.0` — вдвое с
половиной не в ту сторону, и никакое `ℏ` этого не исправит.

ВОПРОС: переворачивает ли сглаживание знак `s_л − s_м`?

ПОЧЕМУ ЭТОТ ВОПРОС СВОБОДЕН ОТ ВЧЕРАШНЕГО ОТЗЫВА. Оба сектора мерятся
ОДНИМ оператором при одном `ε`, поэтому общий множитель нормировки
сокращается в сравнении `s_л` против `s_м`. Отзыв касался абсолютной
величины сглаженного действия; здесь она не нужна. Зависит от
нормировки только ЗНАЧЕНИЕ `ℏ`, и оно отмечено как таковое.

  python3 band.py
"""
import math
import time

import numpy as np

from bd import action
from smear import smeared
from sprinkle import sprinkle_d
from uphill import gen3

K4 = 4.0 / math.sqrt(6.0)
PARTS = (0.25, 0.5, 0.25)
LN2_4 = 0.25 * math.log(2.0)        # ln count_слоистых на N²


def sig(gen, n, eps, seeds, root, **kw):
    v = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 7919 * s), **kw)
        if isinstance(A, tuple):
            A = A[0]
        v.append(action(A, 4)[0] if eps <= 0 else smeared(A, eps, 4))
        del A
    v = np.array(v, float)
    sd = v.std(ddof=1)
    return v.mean(), sd, sd / math.sqrt(2.0 * (len(v) - 1))


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    NS = (800, 1600, 3200)
    EPS = (0.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
    print('=' * 104)
    print('s = σ(S)/N для двух секторов, одним оператором при каждом ε')
    print(f"{'ε':>8}" + ''.join(f'{f"N={n}":>13}' for n in NS)
          + f"{'  s слоист':>12}{'s многообр':>12}{'s_л − s_м':>12}{'знак':>8}")
    rows = []
    for eps in EPS:
        sl, sm = [], []
        cells = []
        for n in NS:
            _, a, _ = sig(gen3, n, eps, 20, 1000 + n, parts=PARTS, q=0.5)
            _, b, _ = sig(sprinkle_d, n, eps, 20, 2000 + n, d=4)
            sl.append(a / n); sm.append(b / n)
            cells.append(f'{a/n:>6.3f}/{b/n:<6.3f}')
        SL, SM = float(np.mean(sl)), float(np.mean(sm))
        tag = 'резкое' if eps <= 0 else f'{eps:g}'
        flag = 'ДА' if SL > SM else 'нет'
        print(f'{tag:>8}' + ''.join(f'{c:>13}' for c in cells)
              + f'{SL:>12.3f}{SM:>12.3f}{SL-SM:>+12.3f}{flag:>8}')
        rows.append((eps, SL, SM))
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print('  (в ячейках: s слоистых / s многообразных при данном N)')
    print()
    print('=' * 104)
    print('ИТОГ')
    good = [r for r in rows if r[1] > r[2]]
    if not good:
        print(f'''
   Ни при одном `ε` слоистые не шумят сильнее многообразных.
   Значит `s_л² − s_м² < 0` при любом сглаживании, условие
   `s_л² − s_м² > 0.346 ℏ²` неразрешимо, и ФАЗОВЫЙ ОТБОР ЗАКРЫТ —
   не из-за величины `ℏ`, а по знаку.

   Причина содержательна: действие БД не самоусредняется на
   сприклингах ОДНОЙ И ТОЙ ЖЕ геометрии (`σ = 2.0·N`), то есть фазы
   физически тождественных конфигураций уже перемешаны между собой.
   Гасить слоистые нечем, потому что многообразные гасят себя сами.
''')
    else:
        print('   ЗНАК ПЕРЕВЁРНУТ при:')
        for eps, SL, SM in good:
            hmax = math.sqrt(max(SL ** 2 - SM ** 2, 0.0) / (2 * LN2_4))
            print(f'      ε = {eps:g}:  s_л = {SL:.3f} > s_м = {SM:.3f},'
                  f'  полоса  ℏ < {hmax:.3f}')
        print()
        print('   (значение `ℏ` зависит от нормировки оператора — см. отзыв')
        print('    в `smear-doc.md`; от неё НЕ зависит сам знак.)')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
