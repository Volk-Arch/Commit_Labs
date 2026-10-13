#!/usr/bin/env python3
"""Горизонт на конусном порядке.

`horizon-doc.md` закрыл горизонты на покомпонентном порядке: выключение
шкал РАЗРЕШАЕТ связи (доля пар 0.1301 / 0.2595 / 0.5036 / 1.0000 при
0/1/2/3 выключенных), а горизонт требует запрещать. И там же назван
необходимый следующий шаг: «чтобы горизонт стал возможен, правило должно
зависеть от направления, а не только от набора шкал».

Направленное правило теперь есть (correlated-scales-doc.md). Горизонт
строится как наклон конуса: добавляем сносовое поле v(x), и условие
причинности становится

    dt > |dx - v(x)*dt|

При |v| < 1 это обычный конус в других координатах. При |v| > 1 конус
опрокидывается целиком, и наружу выйти нельзя — это и есть горизонт.
Так устроены координаты Пенлеве-Гюллстранда для Шварцшильда, где
v = sqrt(r_s/r), и так же устроены звуковые горизонты Унру.

Проверяется:
  1. связи ИЗ-ЗА горизонта наружу — должно быть ноль;
  2. связи СНАРУЖИ внутрь — должно быть много (горизонт односторонний);
  3. ацикличность — опрокинутый конус мог бы дать замкнутые
     времениподобные кривые, и это надо исключить;
  4. геометрия внутри и снаружи по дискриминаторам.

  python3 horizoncone.py
  python3 horizoncone.py --rs 0.3 --n 2000
"""
import argparse
import math
import time

import numpy as np

from core import dimension, frac_pairs, spectral
from majority import has_cycle, shortest_cycle

T_MAX = 2.0
L_BOX = 1.0


def sprinkle_box(N, rs, r_min=0.05):
    """Засев в ящике, кроме окрестности r=0 (там снос расходится)."""
    out = []
    while len(out) < N:
        c = np.empty((2 * N, 4))
        c[:, 0] = rs.uniform(0, T_MAX, 2 * N)
        c[:, 1:] = rs.uniform(-L_BOX, L_BOX, (2 * N, 3))
        ok = np.linalg.norm(c[:, 1:], axis=1) > r_min
        out.extend(c[ok][:N - len(out)])
    return np.array(out[:N])


def drift(x, r_s, cap=4.0):
    """Радиальный снос внутрь: |v| = sqrt(r_s/r), горизонт при r = r_s."""
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    mag = np.minimum(np.sqrt(r_s / np.maximum(r, 1e-9)), cap)
    return -mag * x / np.maximum(r, 1e-9)


def tilted_rel(X, r_s):
    """R[i,j] = True, если j раньше i при наклонённом конусе.

    Снос берётся в пространственной середине пары — простейшее
    согласованное приближение для локального условия.
    """
    dt = X[:, None, 0] - X[None, :, 0]
    dx = X[:, None, 1:] - X[None, :, 1:]
    mid = 0.5 * (X[:, None, 1:] + X[None, :, 1:])
    v = drift(mid, r_s)
    eff = np.linalg.norm(dx - v * dt[..., None], axis=-1)
    R = dt > eff
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def main():
    ap = argparse.ArgumentParser(description='Горизонт на конусном порядке')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--rs', type=float, default=0.3)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n}, радиус горизонта r_s={a.rs}, '
          f'ящик {L_BOX}, время {T_MAX}, семян {a.seeds}')
    print()
    print('1. ОДНОСТОРОННОСТЬ: связи через горизонт')
    print(f"{'r_s':>7}{'внутри':>8}{'снаружи':>9}{'изнутри наружу':>16}"
          f"{'снаружи внутрь':>16}{'ациклично':>11}")
    for r_s in (0.0, 0.1, 0.2, 0.3, 0.5):
        tot_out, tot_in, nin, nout, acy = 0, 0, 0, 0, True
        for s in range(a.seeds):
            rs = np.random.default_rng(100 + 17 * s)
            X = sprinkle_box(a.n, rs)
            R = tilted_rel(X, r_s)
            if has_cycle(R):
                acy = False
            r = np.linalg.norm(X[:, 1:], axis=1)
            inside = r < r_s if r_s > 0 else np.zeros(len(X), bool)
            nin += int(inside.sum())
            nout += int((~inside).sum())
            # R[i,j]: j раньше i. «изнутри наружу» = j внутри, i снаружи
            tot_out += int(R[np.ix_(~inside, inside)].sum())
            tot_in += int(R[np.ix_(inside, ~inside)].sum())
        print(f'{r_s:>7.1f}{nin/a.seeds:>8.0f}{nout/a.seeds:>9.0f}'
              f'{tot_out/a.seeds:>16.1f}{tot_in/a.seeds:>16.1f}'
              f'{("да" if acy else "НЕТ"):>11}')

    print()
    print('2. ГЕОМЕТРИЯ ВНУТРИ И СНАРУЖИ')
    print(f"{'область':>12}{'событий':>9}{'доля пар':>11}"
          f"{'размерность':>16}{'спектр':>11}")
    rs = np.random.default_rng(100)
    X = sprinkle_box(a.n, rs)
    R = tilted_rel(X, a.rs)
    r = np.linalg.norm(X[:, 1:], axis=1)
    for lbl, m in (('внутри', r < a.rs), ('снаружи', r >= a.rs)):
        sub = R[np.ix_(m, m)]
        if m.sum() < 40:
            print(f'{lbl:>12}{int(m.sum()):>9} мало событий')
            continue
        try:
            d = dimension(R=sub)
        except Exception:
            d = float('nan')
        print(f'{lbl:>12}{int(m.sum()):>9}{frac_pairs(sub):>11.4f}'
              f'{d:>16.3f}{spectral(sub):>11.4f}')

    print()
    print('3. ПРОВЕРКА НА ЗАМКНУТЫЕ ВРЕМЕНИПОДОБНЫЕ КРИВЫЕ')
    print('   опрокинутый конус мог бы их дать — надо исключить')
    for r_s in (0.3, 0.5, 1.0):
        bad, short = 0, []
        for s in range(a.seeds):
            rs = np.random.default_rng(400 + 17 * s)
            X = sprinkle_box(a.n, rs)
            R = tilted_rel(X, r_s)
            if has_cycle(R):
                bad += 1
                short.append(shortest_cycle(R))
        sh = min([x for x in short if x > 0], default='-')
        print(f'   r_s={r_s}: с циклом {bad} из {a.seeds}, '
              f'кратчайший {sh}')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
