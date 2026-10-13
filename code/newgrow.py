#!/usr/bin/env python3
"""НОВЫЕ ПРАВИЛА РОСТА: локальность, заданная общим прошлым.

ЧАСТЬ 1 закрывает лазейку в КПР. В `density.py` нашлось, что `S⁴/N`
меняет знак в разреженной области (постоянное `m = 1`: +1.14 при
`d = 6.57`; `m = 2`: −3.00 при `d = 4.26`). Значит где-то между ними
четырёхмерное действие обращается в нуль. Проверка: что там с
обилиями? Оба прибора в одних точках; если нулю действия отвечает
провал обилий, ни одна связь КПР не проходит обе проверки.

ЧАСТЬ 2 — новая теория. Известно, ПОЧЕМУ КПР не даёт многообразий, и
причина, похоже, в НЕЛОКАЛЬНОСТИ правила: у Райдаута и Соркина новый
элемент берёт `m` предков РАВНОМЕРНО среди всех, а в сприклинге звенья
элемента идут к локализованному множеству возле его светового конуса.
Локальность выражается чисто порядком: два максимальных элемента
близки, если у них ОБЩЕЕ ПРОШЛОЕ.

    L1 «поверхность»  — предки берутся среди МАКСИМАЛЬНЫХ элементов
                        (текущего «настоящего»), равномерно;
    L2 «локальная»    — выбирается опора среди максимальных, остальные
                        предки — те максимальные, у кого прошлое
                        наиболее ОБЩЕЕ с опорой;
    L3 «мягкая»       — то же, но вероятность ∝ (перекрытие)^β, то есть
                        локальность с настраиваемой жёсткостью.

L1 уже отличается от КПР тем, что звенья идут только к поверхности.
L2 добавляет пространственную локализацию. Ни то, ни другое в
литературе не встречалось: там рассматриваются только связи,
зависящие от |S|, что по построению нелокально.

ОГОВОРКА, сразу. L2 и L3 НАРУШАЮТ общую ковариацию Райдаута–Соркина
(выбор зависит не только от числа предков), поэтому они не являются
КПР-динамиками и на вывод про КПР не влияют. Это проверка другого
вопроса: может ли ЛОКАЛЬНОЕ правило роста давать многообразие вообще.

  python3 newgrow.py
"""
import math
import time

import numpy as np

from grow import mm_dim, c_k, gen_sprinkle, intervals, spectrum
from csg import gen_csg
from bd import action
from density import iv_rows


def gen_local(n, m, rs, mode='surface', beta=2.0):
    """Рост на поверхности. mode: surface | local | soft."""
    A = np.zeros((n, n), dtype=bool)
    ismax = np.zeros(n, dtype=bool)
    ismax[0] = True
    for j in range(1, n):
        mx = np.flatnonzero(ismax[:j])
        if len(mx) == 0:
            mx = np.array([j - 1])
        k = min(int(m), len(mx))
        if mode == 'surface' or len(mx) <= k:
            S = rs.choice(mx, size=k, replace=False)
        else:
            a = int(mx[rs.integers(len(mx))])
            ov = (A[:, mx] & A[:, a][:, None]).sum(axis=0).astype(float)
            ov[mx == a] = np.inf
            if mode == 'local':
                S = mx[np.argsort(-ov)[:k]]
            else:
                w = (ov + 1.0) ** beta
                w[~np.isfinite(w)] = w[np.isfinite(w)].max() * 1e3 if np.isfinite(w).any() else 1.0
                w = w / w.sum()
                S = rs.choice(mx, size=k, replace=False, p=w)
        col = np.zeros(n, dtype=bool)
        col[S] = True
        col[:j] |= A[:j, S].any(axis=1)
        A[:, j] = col
        ismax[S] = False
        ismax[j] = True
    return A


def both(gen, n, seeds, root, lo=60, hi=300, want=30, **kw):
    """Оба прибора: обилия интервалов и действие БД на ТЕХ ЖЕ интервалах."""
    P = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 17 * s), **kw)
        a2, _, _ = action(A, 2)
        a4, _, _ = action(A, 4)
        r = float(np.count_nonzero(A)) / (n * (n - 1) / 2.0)
        dall = mm_dim(r / 2.0) if 0 < r < 1 else np.nan
        rows = intervals(A, want, lo, hi, np.random.default_rng(root + 3 + s))
        sub = iv_rows(A, want, lo, hi, np.random.default_rng(root + 3 + s), 4)
        del A
        if not rows or not sub:
            continue
        R = np.array(rows, float)
        c1 = R[:, 0]
        x2 = float((R[:, 1] / (c1 * (c1 - 1))).mean())
        div = mm_dim(x2)
        SS = np.array(sub, float)
        if not np.isfinite(div) or div <= 0.3:
            continue
        p3 = c_k(3, div) * (c1 * (c1 - 1) * (c1 - 2))
        p4 = c_k(4, div) * (c1 * (c1 - 1) * (c1 - 2) * (c1 - 3))
        P.append((dall, a2 / n, a4 / n, div,
                  float((R[:, 2] / p3).mean()), float((R[:, 3] / p4).mean()),
                  float((SS[:, 2] / SS[:, 0]).mean())))
    if len(P) < 2:
        return None
    P = np.array(P, float)
    return P.mean(axis=0), P.std(axis=0, ddof=1) / math.sqrt(len(P)), len(P)


HDR = (f"{'что':>30}{'d всего':>9}{'S²/N':>9}{'S⁴/N':>9}"
       f"{'d интерв':>14}{'C₃/пред':>14}{'C₄/пред':>14}{'S⁴/n инт':>12}")


def line(tag, r):
    if r is None:
        print(f'{tag:>30}   интервалов мало')
        return
    m, e, k = r
    print(f'{tag:>30}{m[0]:>9.2f}{m[1]:>9.2f}{m[2]:>9.2f}'
          f'{m[3]:>8.2f}±{e[3]:<5.2f}{m[4]:>8.3f}±{e[4]:<5.3f}'
          f'{m[5]:>8.3f}±{e[5]:<5.3f}{m[6]:>12.2f}')


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N = 1200
    print('=' * 112)
    print('ЭТАЛОН')
    print(HDR)
    for dd in (2, 4):
        line(f'сприклинг {dd}D',
             both(gen_sprinkle, N, 8, 200 + dd, d=dd))
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 112)
    print('ЧАСТЬ 1. КПР в разреженной области, где S⁴/N меняет знак')
    print(HDR)
    for m in (1, 2, 3, 5):
        line(f'постоянное m={m}',
             both(gen_csg, N, 6, 3100 + m,
                  draw_m=(lambda j, rs, m=m: m)))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    for mu in (1.2, 1.6, 2.5):
        line(f'Пуассон mu={mu}',
             both(gen_csg, N, 6, 3300 + int(100 * mu),
                  draw_m=(lambda j, rs, mu=mu: max(1, int(rs.poisson(mu))))))
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 112)
    print('ЧАСТЬ 2. НОВЫЕ ПРАВИЛА: рост на поверхности и локальный')
    print(HDR)
    for m in (2, 4, 8, 16, 32):
        line(f'L1 поверхность m={m}',
             both(gen_local, N, 6, 5000 + m, m=m, mode='surface'))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    for m in (4, 8, 16, 32):
        line(f'L2 локальный m={m}',
             both(gen_local, N, 5, 5200 + m, m=m, mode='local'))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    for m, b in ((8, 1.0), (8, 4.0), (16, 1.0), (16, 4.0)):
        line(f'L3 мягкий m={m} b={b}',
             both(gen_local, N, 5, 5400 + m + int(10 * b), m=m,
                  mode='soft', beta=b))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
