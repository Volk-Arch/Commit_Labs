#!/usr/bin/env python3
"""ДВЕРЬ ВОГНУТОСТИ: разделяет ли ШИРИНА.

Перепись (`census-doc.md`) закрыла все счётные двери и оставила
вогнутость: строго вогнутая функция МОЖЕТ иметь строгий максимум во
внутренней точке выпуклой оболочки, линейная не может. А замер ширины
(`compose-doc.md`) дал конкретный вид: ширина = максимальная антицепь =
граница = пространство. Она НЕЛИНЕЙНА и НЕЭКСТЕНСИВНА (максимум по
антицепям), то есть вне класса, закрытого Фаркашем.

Различитель замерен: отношение ширина/N падает у засева (0.475 ->
0.355) и стоит у слоистого врага (0.340 -> 0.334).

ТОНКОСТЬ ПОСТАНОВКИ. При фиксированном N ширина есть просто число, а
различитель сидит в ПОКАЗАТЕЛЕ. Поэтому координата `ширина/N` даёт
ровно то же, что `ширина` (деление на постоянную не меняет ЛП), а
`log(ширина)` — уже другое, потому что нелинейное преобразование меняет
разности непропорционально. Берутся обе, и это записано, чтобы не
обмануться.

ЯКОРЬ: восемь объёмных счётов обязаны НЕ разделять (сегодня: -1308 на
широком наборе, сертификат действителен). Без воспроизведения якоря
остальные строки не считаются.

Соперники генерируются СОСТЯЗАТЕЛЬНО: сетка дала ложную выполнимость
уже дважды за день.
"""
import math

import numpy as np

from census1 import feats, sprinkle_rc
from census2 import certify, make, solve, verdict
from census3 import perc, sprink_d
from width3 import width


def enrich(m, n):
    """К счётам соперника добавить ширину и её логарифм."""
    o, ab, b, S, tag = m[0], m[1], m[2], m[3], m[4]
    return o, ab, b, S, tag, m[5]


def wfeat(Rc, n):
    w = float(width(Rc))
    return np.array([w]), np.array([math.log(max(w, 1.0))])


def main():
    print(__doc__)
    n, rs = 400, np.random.default_rng(999331)
    S4 = 4.0 * n * math.log(n)

    Rc4 = sprinkle_rc(n, rs)
    o4, ab4, b4, _ = feats(Rc4, n)
    w4, lw4 = wfeat(Rc4, n)
    print(f'N = {n}; ширина засева {w4[0]:.0f} '
          f'(отношение {w4[0]/n:.3f}), энтропия 4N lnN = {S4:.0f}')

    def build(gen_rc, m):
        """m = (o, ab, b, S, tag); добавить ширину из его матрицы."""
        return m

    pool = []

    def add_layered(k, q, kind):
        mm = make(n, k, q, kind, rs)
        if mm is None:
            return None
        # make() не отдаёт матрицу; пересобираю её тем же путём
        return mm

    # соперники с матрицами, чтобы считать ширину
    from census2 import layered_S, profiles

    def mk(k, q, kind):
        s = profiles(n, k, rs, kind)
        if s is None:
            return None
        Rc, S = layered_S(n, s, q, rs)
        o, ab, b, _ = feats(Rc, n)
        w, lw = wfeat(Rc, n)
        return o, ab, b, S, (k, round(q, 3), kind), w, lw

    for k in (2, 3, 4, 5, 6, 8, 11, 15, 21, 30, 45, 70, 110, 180):
        for q in (0.03, 0.07, 0.13, 0.22, 0.35, 0.5, 0.7, 0.9):
            for kind in (0, 1, 2):
                m = mk(k, q, kind)
                if m:
                    pool.append(m)
    nl = len(pool)
    for p in (0.004, 0.01, 0.025, 0.06, 0.15, 0.4):
        m = perc(n, p, rs)
        Rcp = None
        # перколяцию пересоберём, чтобы взять ширину
        R = np.triu(rs.random((n, n)) < p, 1)
        from census1 import closure as cl
        Rcp = cl(R)
        o, ab, b, _ = feats(Rcp, n)
        w, lw = wfeat(Rcp, n)
        M = n * (n - 1) / 2
        h2 = (-p * math.log(p) - (1 - p) * math.log1p(-p)) if 0 < p < 1 else 0.
        pool.append((o, ab, b, M * h2, ('перк', p, '-'), w, lw))
    for d in (2, 3, 5, 6):
        for _ in range(3):
            from sectors import sprinkle_d
            Rd = sprinkle_d(n, d, rs)
            o, ab, b, _ = feats(Rd, n)
            w, lw = wfeat(Rd, n)
            pool.append((o, ab, b, S4, ('засев', d, '-'), w, lw))
    print(f'соперников: слоистых {nl}, всего {len(pool)}')

    sets = [
        ('ЯКОРЬ: восемь объёмных', lambda m: m[0], o4),
        ('+ ШИРИНА', lambda m: np.concatenate([m[0], m[5]]),
         np.concatenate([o4, w4])),
        ('+ log(ШИРИНА)', lambda m: np.concatenate([m[0], m[6]]),
         np.concatenate([o4, lw4])),
        ('+ ширина и log(ширина)',
         lambda m: np.concatenate([m[0], m[5], m[6]]),
         np.concatenate([o4, w4, lw4])),
        ('+ граничный + ширина',
         lambda m: np.concatenate([m[0], m[2], m[5]]),
         np.concatenate([o4, b4, w4])),
    ]
    for name, pick, f4 in sets:
        cur = list(pool)
        sc = np.maximum(np.abs(f4), 1.0)
        out = None
        for it in range(7):
            G = np.array([(f4 - pick(m)) / sc for m in cur])
            C = np.array([m[3] for m in cur]) - S4
            t, th = solve(G, C)
            if t is None or t <= 0:
                out = (t, it, len(cur), G, C, cur)
                break
            found = []
            for _ in range(90):
                k = int(rs.integers(2, 200))
                q = float(10 ** rs.uniform(-2.3, -0.004))
                m = mk(k, q, int(rs.integers(0, 3)))
                if m is None:
                    continue
                if float(th @ ((f4 - pick(m)) / sc)) - (m[3] - S4) <= 0:
                    found.append(m)
            if not found:
                out = (t, it, len(cur), None, None, cur)
                break
            cur += found[:16]
        t, it, ncur, G, C, cur = out
        print(f'\n{name} ({len(f4)} координат, круг {it}, '
              f'соперников {ncur})')
        print(f'  {verdict(t)}')
        if t is not None and t <= 0 and G is not None:
            c = certify(G, C)
            if c:
                res, sl, lam = c
                print(f'  Фаркаш: |Gᵗλ|={res:.1e}, Σλ·C={sl:+.2f} => '
                      f'{"ДЕЙСТВИТЕЛЕН" if res<1e-6 and sl>0 else "нет"}')
                top = np.argsort(lam)[::-1][:3]
                print('  смесь: ' + '; '.join(
                    f'λ={lam[i]:.3f} {cur[i][4]} ширина {cur[i][5][0]:.0f}'
                    for i in top if lam[i] > 1e-4))
                wmix = float(sum(lam[i] * cur[i][5][0]
                                 for i in range(len(cur))))
                print(f'  ширина смеси {wmix:.1f} против засева '
                      f'{w4[0]:.0f} — разность {w4[0]-wmix:+.1f}')
        else:
            print('  нарушителей при состязательном поиске не нашлось')


if __name__ == '__main__':
    main()
