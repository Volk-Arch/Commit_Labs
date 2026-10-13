#!/usr/bin/env python3
"""РАЗМЕРНОСТЬ СЕМЕЙСТВ — прибор на ПАРНЫХ разностях.

ЧТО БЫЛО НЕ ТАК в `dimobs3.py`: контроль провалился. Нулевой шаг по
параметру дал ранг 1 с сингулярным числом 50 — прибор видел
направление там, где его нет. Причина: возмущённая партия считалась на
ДРУГИХ засевах (`root + 555`), и разность несла не только отклик, но и
шум между двумя независимыми партиями; на корзинах с почти нулевой
дисперсией он и вылез. Это ровно ошибка «сравнение по разным мирам
вместо парного», на которой проект уже попадался.

ПОЧИНКА, одна на оба дефекта: отклик мерится ПАРНО, на одном и том же
засеве.

    Delta_s = obs(theta + delta, rng_s) - obs(theta, rng_s)

Тогда нулевой шаг даёт тождественный нуль (контроль становится
точным), а межпартийный шум уходит целиком.

Для спаривания слоистый генератор переписан: поле случайности тянется
ОДИН раз на весь порядок и потом нарезается по слоям. В старом
`rs.random` брал матрицу формы, зависящей от `parts`, поэтому сдвиг
доли слоя менял число потреблённых чисел и расцеплял поток для всех
последующих слоёв.

Знаменатель — шум ОДНОЙ реализации (различимость в одной вселенной, а
не в среднем по восьми), с полом в один отсчёт: счётчик не может
дрогнуть меньше, чем на единицу.

Второй контроль — порог значимости на каждую клетку: клетки с
|средний отклик| < 2 sigma / sqrt(засевов) обнуляются. Это может ранг
только понизить.

  python3 dimobs4.py
"""
import math
import time

import numpy as np

import dimobs2 as D
from dimobs import gen_conf


def gen_layers_p(n, parts, qs, rs):
    """Слоистый порядок, СПАРИВАЕМЫЙ: поле случайности одно на весь
    порядок, нарезается по слоям, поэтому сдвиг долей не расцепляет
    поток."""
    U = rs.random((n, n))
    p = np.abs(np.array(parts, float)); p /= p.sum()
    sz = np.maximum(np.floor(p * n).astype(int), 1)
    sz[-1] += n - sz.sum()
    sz = np.maximum(sz, 1)
    idx, c = [], 0
    for s in sz:
        idx.append(np.arange(c, min(c + s, n))); c += s
    A = np.zeros((n, n), dtype=bool)
    M = []
    for i in range(len(idx) - 1):
        q = min(max(float(qs[i]), 0.0), 1.0)
        m = U[np.ix_(idx[i], idx[i + 1])] < q
        A[np.ix_(idx[i], idx[i + 1])] = m
        M.append(m)
    for dd in range(2, len(idx)):
        for i in range(len(idx) - dd):
            acc = M[i].astype(np.float32)
            for j in range(i + 1, i + dd):
                acc = acc @ M[j].astype(np.float32)
                np.minimum(acc, 1.0, out=acc)   # «есть путь», не число путей
            A[np.ix_(idx[i], idx[i + dd])] = acc > 0.5
    return A


def paired(gen, n, th0, steps, seeds, root):
    """J[i,j] = <Delta_ij> / sigma_i, парно по засевам."""
    flat, shape = D.flat_pack(th0)
    base, dif = [], [[] for _ in steps]
    for s in range(seeds):
        sd = root + 7919 * s
        b = D.obs(gen(n, *th0, np.random.default_rng(sd)))
        base.append(b)
        for j, st in enumerate(steps):
            f2 = list(flat); f2[j] += st
            th = D.unpack(f2, shape)
            a = D.obs(gen(n, *th, np.random.default_rng(sd)))
            dif[j].append(a - b)
    B = np.array(base, float)
    sig = np.maximum(B.std(axis=0, ddof=1), 1.0 / n)
    cols, gated = [], 0
    for j in range(len(steps)):
        Dj = np.array(dif[j], float)
        mu = Dj.mean(axis=0)
        er = Dj.std(axis=0, ddof=1) / math.sqrt(seeds)
        bad = np.abs(mu) < 2.0 * er
        gated += int(bad.sum())
        mu = np.where(bad, 0.0, mu)
        cols.append(mu / sig)
    J = np.column_stack(cols)
    keep = np.abs(J).max(axis=1) > 1e-12
    J = J[keep]
    if J.size == 0:
        return np.zeros(1), 0, 0, gated
    sv = np.linalg.svd(J, compute_uv=False)
    return sv, int((sv > 1.0).sum()), int(keep.sum()), gated


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N, SD = 1100, 8
    K, L = 24, 16
    parts, qs = [1.0 / L] * L, [0.5] * (L - 1)
    lst = [0.03] * L + [0.05] * (L - 1)
    print(f'N = {N}, засевов {SD}; слоистое L = {L} ({2*L-1} парам.), '
          f'метрика K = {K} ({K} парам.)')
    print()
    print('=' * 72)
    print('КОНТРОЛЬ 1: нулевой шаг. Парность обязана дать ранг РОВНО 0.')
    for nm, gen, th, m in (('метрика ', gen_conf, ([0.0] * 6,), 6),
                           ('слоистое', gen_layers_p,
                            ([1 / 6.] * 6, [0.5] * 5), 11)):
        D.NDEP = 12
        sv, r, nk, g = paired(gen, N, th, [0.0] * m, SD, 777)
        print(f'  {nm}: ранг {r}, максимум {sv.max():.2e}  '
              f'-> {"ЧИСТО" if r == 0 else "ПРИБОР ВРЁТ"}')
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 72)
    print('СКАН ПО РАЗРЕШЕНИЮ: чей ранг растёт с числом полос глубины?')
    print(f"{'полос':>7}{'компонент':>11}{'метрика':>10}{'слоистое':>11}"
          f"{'обнулено клеток':>18}")
    rows = []
    for nd in (3, 6, 12, 24, 40):
        D.NDEP = nd
        _, rc, _, gc = paired(gen_conf, N, ([0.0] * K,), [0.04] * K,
                              SD, 9100 + nd)
        D.NDEP = nd
        _, rl, _, gl = paired(gen_layers_p, N, (parts, qs), lst,
                              SD, 9200 + nd)
        rows.append((nd, rc, rl))
        print(f'{nd:>7}{7*nd:>11}{rc:>10}{rl:>11}'
              f'{f"{gc}/{gl}":>18}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 72)
    nd0, rc0, rl0 = rows[0]
    nd1, rc1, rl1 = rows[-1]
    print(f'метрика:  {rc0} -> {rc1} из {K} при росте разрешения в '
          f'{nd1//nd0} раз')
    print(f'слоистое: {rl0} -> {rl1} из {2*L-1}')


if __name__ == '__main__':
    main()
