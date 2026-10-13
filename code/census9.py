#!/usr/bin/env python3
"""ЖЁСТКАЯ ПРОВЕРКА положительных вердиктов census8.

census8 дал: восемь объёмных НЕ разделяют (якорь, -2665); ширина сама
по себе НЕ помогает (смесь воспроизводит её РОВНО: 182.0 против 182);
а разделяют только пары — (ширина, log ширины) и (граница, ширина).

Это подозрительно именно так: пара (w, log w) позволяет действию
выцелить КОНКРЕТНОЕ значение ширины, а не показатель, то есть выучить
один образец. И за день два моих «разделяет» уже сломались при более
сильном поиске.

ТРИ ПРОВЕРКИ, каждая способна убить вывод:
  1. ПОДМЕНА. То же theta на 10 СВЕЖИХ засевах d=4, которых не было
     при построении. Если свежий засев получает отрицательный запас,
     действие выучило выборку, а не класс. Это главная проверка.
  2. ПРИЦЕЛЬНЫЙ СКАН в известном опасном семействе: k = 3…7, тонкие
     края (1, 2, 3 элемента), тонкая сетка по q от 0.02 до 0.5, по
     три зерна. Случайный поиск там редко попадает.
  3. ДРУГОЙ N. Действие, найденное при N=400, проверяется на засеве и
     соперниках при N=600: настоящее правило не должно зависеть от
     размера.
"""
import math

import numpy as np

from census1 import feats, sprinkle_rc
from census2 import layered_S, profiles, solve, verdict
from census3 import perc, sprink_d
from width3 import width


def wf(Rc):
    w = float(width(Rc))
    return np.array([w]), np.array([math.log(max(w, 1.0))])


def mk(n, k, q, kind, rs, thin=None):
    if thin is None:
        s = profiles(n, k, rs, kind)
    else:
        base = n // k
        s = [base] * k
        s[-1] += n - base * k
        if k >= 3 and base > thin:
            s[0] = s[-1] = thin
            s[k // 2] += n - sum(s)
        if min(s) < 1 or sum(s) != n:
            return None
    if s is None:
        return None
    Rc, S = layered_S(n, s, q, rs)
    o, ab, b, _ = feats(Rc, n)
    w, lw = wf(Rc)
    return o, ab, b, S, (k, round(q, 3), kind if thin is None else f't{thin}'), w, lw


def pool_for(n, rs, S4):
    pool = []
    for k in (2, 3, 4, 5, 6, 8, 11, 15, 21, 30, 45, 70, 110, 180):
        for q in (0.03, 0.07, 0.13, 0.22, 0.35, 0.5, 0.7, 0.9):
            for kind in (0, 1, 2):
                m = mk(n, k, q, kind, rs)
                if m:
                    pool.append(m)
    from census1 import closure as cl
    for p in (0.004, 0.01, 0.025, 0.06, 0.15, 0.4):
        Rcp = cl(np.triu(rs.random((n, n)) < p, 1))
        o, ab, b, _ = feats(Rcp, n)
        w, lw = wf(Rcp)
        M = n * (n - 1) / 2
        h2 = (-p * math.log(p) - (1 - p) * math.log1p(-p))
        pool.append((o, ab, b, M * h2, ('перк', p, '-'), w, lw))
    from sectors import sprinkle_d
    for d in (2, 3, 5, 6):
        for _ in range(3):
            Rd = sprinkle_d(n, d, rs)
            o, ab, b, _ = feats(Rd, n)
            w, lw = wf(Rd)
            pool.append((o, ab, b, S4, ('засев', d, '-'), w, lw))
    return pool


def main():
    print(__doc__)
    n, rs = 400, np.random.default_rng(606060)
    S4 = 4.0 * n * math.log(n)
    Rc4 = sprinkle_rc(n, rs)
    o4, ab4, b4, _ = feats(Rc4, n)
    w4, lw4 = wf(Rc4)
    pool = pool_for(n, rs, S4)

    cases = [
        ('(ширина, log ширины)',
         lambda m: np.concatenate([m[0], m[5], m[6]]),
         np.concatenate([o4, w4, lw4])),
        ('(граница, ширина)',
         lambda m: np.concatenate([m[0], m[2], m[5]]),
         np.concatenate([o4, b4, w4])),
    ]
    for name, pick, f4 in cases:
        sc = np.maximum(np.abs(f4), 1.0)
        cur = list(pool)
        th = None
        for it in range(5):
            G = np.array([(f4 - pick(m)) / sc for m in cur])
            C = np.array([m[3] for m in cur]) - S4
            t, th = solve(G, C)
            if t is None or t <= 0:
                break
            bad = []
            for _ in range(70):
                m = mk(n, int(rs.integers(2, 200)),
                       float(10 ** rs.uniform(-2.3, -0.004)),
                       int(rs.integers(0, 3)), rs)
                if m and float(th @ ((f4 - pick(m)) / sc)) - (m[3] - S4) <= 0:
                    bad.append(m)
            if not bad:
                break
            cur += bad[:14]
        print(f'\n=== {name}: {verdict(t)} ===')
        if t is None or t <= 0:
            print('  сломалось ещё до проверок')
            continue

        print('  ПРОВЕРКА 1 (подмена): свежие засевы d=4')
        vals = []
        for _ in range(10):
            R = sprinkle_rc(n, rs)
            o, ab, b, _ = feats(R, n)
            w, lw = wf(R)
            mm = (o, ab, b, S4, ('свежий', 4, '-'), w, lw)
            vals.append(float(th @ ((f4 - pick(mm)) / sc)))
        vals = np.array(vals)
        print('    запасы: ' + ', '.join(f'{v:+.3g}' for v in vals))
        print(f'    отрицательных {int((vals<0).sum())}/10 — '
              + ('ДЕЙСТВИЕ ВЫУЧИЛО ВЫБОРКУ' if (vals < 0).sum() >= 3
                 else 'переносится на свежие засевы'))

        print('  ПРОВЕРКА 2 (прицельный скан опасного семейства)')
        worst, warg = math.inf, None
        for k in (3, 4, 5, 6, 7):
            for thin in (1, 2, 3, None):
                for q in np.arange(0.02, 0.52, 0.03):
                    for _ in range(2):
                        m = mk(n, k, float(q), 1, rs, thin=thin)
                        if m is None:
                            continue
                        v = (float(th @ ((f4 - pick(m)) / sc))
                             - (m[3] - S4))
                        if v < worst:
                            worst, warg = v, m[4]
        print(f'    минимальный запас {worst:+.4g} при {warg} — '
              + ('СЛОМАНО' if worst <= 0 else 'держится'))

        print('  ПРОВЕРКА 3 (другой N = 600)')
        n2 = 600
        S42 = 4.0 * n2 * math.log(n2)
        R2 = sprinkle_rc(n2, rs)
        o2, ab2, b2, _ = feats(R2, n2)
        w2, lw2 = wf(R2)
        f42 = (np.concatenate([o2, w2, lw2]) if 'log' in name
               else np.concatenate([o2, b2, w2]))
        sc2 = np.maximum(np.abs(f42), 1.0)
        bad2 = 0
        mins = math.inf
        for k in (3, 4, 5, 6):
            for q in (0.05, 0.1, 0.2, 0.35):
                m = mk(n2, k, q, 1, rs)
                if m is None:
                    continue
                v = float(th @ ((f42 - pick(m)) / sc2)) - (m[3] - S42)
                mins = min(mins, v)
                if v <= 0:
                    bad2 += 1
        print(f'    нарушителей {bad2}/16, минимальный запас '
              f'{mins:+.4g} — '
              + ('не переносится на другой N' if bad2 else
                 'переносится'))


if __name__ == '__main__':
    main()
