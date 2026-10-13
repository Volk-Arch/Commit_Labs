#!/usr/bin/env python3
"""ПЕРЕПИСЬ, итог: видит ли найденное действие РАЗМЕРНОСТЬ.

Состояние. При широком наборе соперников (слоистые со всеми профилями,
перколяция, засевы d=2,3,5,6) разделяют только координаты
«распределение размеров интервалов по лог-полосам + граничные счёты».
Пустых полос нет, то есть это не артефакт носителя.

Лемма, делающая весь довод Фаркаша законным: энтропия слоистого
сектора `Σ a_i a_{i+1} H2(q)` ТОЧНА, а не завышена. Замыкание не может
добавить отношение между СОСЕДНИМИ слоями (между ними ничего не
лежит), значит соседняя матрица восстанавливается из порядка, и
отображение матрица -> порядок инъективно.

Здесь проверяется то, на чём сломался граничный счёт в архиве:
«с граничным счётом разделимой становится КАЖДАЯ размерность, значит
четвёрка задаётся коэффициентами, а не выводится». Теперь засевы
d=2,3,5,6 стоят В НАБОРЕ СОПЕРНИКОВ с той же энтропией 4N lnN, то есть
требование к ним сводится к чистому разделению счётов.

Три проверки:
  1. состязательный поиск по слоистым — выживет ли разделение;
  2. запасы ПО СЕМЕЙСТВАМ: если у засевов d=3 и d=5 запас велик, то
     действие видит размерность; если он на нуле — не видит;
  3. КОНТРОЛЬ НА ПОДМЕНУ: то же действие проверяется на засеве d=4 с
     ДРУГИМ зерном, которого не было при построении. Если новый засев
     d=4 получает отрицательный запас, значит действие выучило
     конкретную выборку, а не размерность.
"""
import math

import numpy as np

from census1 import feats, sprinkle_rc
from census2 import certify, make, solve, verdict
from census3 import perc, sprink_d


def main():
    print(__doc__)
    n, rs = 400, np.random.default_rng(161803398)
    S4 = 4.0 * n * math.log(n)
    o4, ab4, b4, _ = feats(sprinkle_rc(n, rs), n)
    f4 = np.concatenate([ab4, b4])
    sc = np.maximum(np.abs(f4), 1.0)

    def vec(m):
        return np.concatenate([m[1], m[2]])

    pool, fam = [], []
    for k in (2, 3, 4, 6, 9, 14, 22, 34, 55, 90, 140, 220):
        for q in (0.03, 0.08, 0.2, 0.45, 0.75, 0.95):
            for kind in (0, 1, 2):
                m = make(n, k, q, kind, rs)
                if m:
                    pool.append(m); fam.append('слоистые')
    for p in (0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5):
        pool.append(perc(n, p, rs)); fam.append('перколяция')
    for d in (2, 3, 5, 6):
        for _ in range(4):
            pool.append(sprink_d(n, d, rs, S4)); fam.append(f'засев d={d}')

    th = None
    for it in range(8):
        G = np.array([(f4 - vec(m)) / sc for m in pool])
        C = np.array([m[3] for m in pool]) - S4
        t, th = solve(G, C)
        if t is None or t <= 0:
            print(f'\nкруг {it}: {verdict(t)} — разделение сломано')
            c = certify(G, C)
            if c:
                res, sl, lam = c
                print(f'  Фаркаш: |Gᵗλ|={res:.1e}, Σλ·C={sl:+.2f} => '
                      f'{"ДЕЙСТВИТЕЛЕН" if res<1e-6 and sl>0 else "нет"}')
                top = np.argsort(lam)[::-1][:3]
                print('  смесь: ' + '; '.join(
                    f'λ={lam[i]:.3f} {fam[i]} {pool[i][4]}'
                    for i in top if lam[i] > 1e-4))
            return
        add = []
        for _ in range(110):
            k = int(rs.integers(2, 200))
            q = float(10 ** rs.uniform(-2.3, -0.004))
            m = make(n, k, q, int(rs.integers(0, 3)), rs)
            if m is None:
                continue
            if float(th @ ((f4 - vec(m)) / sc)) - (m[3] - S4) <= 0:
                add.append(m)
        print(f'круг {it}: соперников {len(pool)}, {verdict(t)}, '
              f'нарушителей {len(add)}/110', flush=True)
        if not add:
            break
        pool += add[:18]
        fam += ['слоистые'] * len(add[:18])

    print('\nПРОВЕРКА 2: запасы по семействам (положительный = '
          'засев d=4 выигрывает)')
    G = np.array([(f4 - vec(m)) / sc for m in pool])
    C = np.array([m[3] for m in pool]) - S4
    marg = G @ th - C
    for f in sorted(set(fam)):
        ix = [i for i, x in enumerate(fam) if x == f]
        mm = marg[ix]
        print(f'  {f:>14}: {len(ix):>3} шт., запас от {mm.min():+.4g} '
              f'до {mm.max():+.4g}')

    print('\nПРОВЕРКА 3 (на подмену): новые засевы d=4, НЕ участвовавшие '
          'в построении')
    bad = 0
    for i in range(8):
        o, ab, b, _ = feats(sprinkle_rc(n, rs), n)
        v = np.concatenate([ab, b])
        g = (f4 - v) / sc
        val = float(th @ g)
        if val < 0:
            bad += 1
        print(f'  засев {i}: theta·(f_обр - f_нов) = {val:+.4g}'
              f'{"   <-- новый ЛУЧШЕ образца" if val < 0 else ""}')
    print(f'  образец проигрывает новым засевам в {bad}/8 случаях — '
          f'{"действие выучило выборку" if bad >= 4 else "разделение не на выборке"}')
    print('\n  (для сравнения: минимальный запас против соперников '
          f'{marg.min():+.4g}; если разброс по своим же засевам того же '
          'порядка, различение мнимое)')


if __name__ == '__main__':
    main()
