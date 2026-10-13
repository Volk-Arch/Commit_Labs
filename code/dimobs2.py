#!/usr/bin/env python3
"""РАЗМЕРНОСТЬ СЕМЕЙСТВА, а не наблюдаемой. Две починки.

ЧТО БЫЛО НЕ ТАК в `dimobs.py`.

  1. У слоистых вышли сингулярные числа 1e14…1e16 — это НЕ информация,
     а деление нуля на нуль: у слоистого порядка населены две корзины из
     тринадцати, остальные тождественно нулевые, и дисперсия, и отклик
     там нули. Починка: компоненты с ничтожной дисперсией И ничтожным
     откликом отбрасываются, а на шум ставится пол в один отсчёт.

  2. ГЛАВНОЕ: мерилась размерность НАБЛЮДАЕМОЙ, а не семейства. У
     конформной метрики различимых направлений вышло 1, 1, 4, 6, 6 при
     K = 1, 2, 4, 8, 16 — упирается в 6, потому что у наблюдаемой всего
     13 компонент. Больше она увидеть не может в принципе.

     А моды конформной метрики ЛОКАЛЬНЫ по `eta`. Значит наблюдаемая
     обязана разрешать время — и такая есть, внутренняя, без координат:
     ПРИЧИННАЯ ГЛУБИНА (длина самой длинной цепи от минимальных
     элементов). Спектр обилий ПО ПОЛОСАМ ГЛУБИНЫ имеет размерность,
     растущую с `N`, и моды различает.

ВОПРОС тот же: растёт ли число различимых направлений с числом
параметров — у слоистых и у метрики. Логарифм числа классов идёт как
`размерность x ln(точность)`, значит размерность и решает.

  python3 dimobs2.py
"""
import math
import time

import numpy as np

from dimobs import gen_layers, gen_conf

NDEP = 6          # полос по причинной глубине
UED = np.array([0.0, 1e-9, 0.01, 0.05, 0.15, 0.40, 1e9])


def depth(A):
    """Причинная глубина каждого элемента: длина самой длинной цепи от
    минимальных. Порядок уже топологически отсортирован по построению
    (столбцы по возрастанию), поэтому одним проходом."""
    n = A.shape[0]
    h = np.zeros(n, dtype=np.int32)
    for j in range(n):
        pre = np.flatnonzero(A[:j, j])
        if len(pre):
            h[j] = h[pre].max() + 1
    return h


def obs(A):
    """Спектр обилий интервалов ПО ПОЛОСАМ ГЛУБИНЫ — наблюдаемая, чья
    размерность растёт с разрешением и которая локальна по времени."""
    n = A.shape[0]
    h = depth(A)
    hm = max(int(h.max()), 1)
    hb = np.minimum((h * NDEP) // (hm + 1), NDEP - 1)
    F = A.astype(np.float32)
    B = np.rint(F @ F).astype(np.int64)
    ii, jj = np.nonzero(A)
    u = B[ii, jj] / float(n)
    ub = np.searchsorted(UED[1:], u, side='right')
    ub = np.minimum(ub, len(UED) - 2)
    key = hb[ii] * (len(UED) - 1) + ub
    out = np.bincount(key, minlength=NDEP * (len(UED) - 1)).astype(float)
    dens = np.bincount(hb, minlength=NDEP).astype(float)
    return np.concatenate([out / n, dens / n])


def flat_pack(th):
    flat, shape = [], []
    for x in th:
        if isinstance(x, (list, tuple, np.ndarray)):
            flat += list(x); shape.append(len(x))
        else:
            flat.append(x); shape.append(0)
    return flat, shape


def unpack(flat, shape):
    out, c = [], 0
    for s in shape:
        if s:
            out.append(flat[c:c + s]); c += s
        else:
            out.append(flat[c]); c += 1
    return out


def sample(gen, n, th, seeds, root):
    V = []
    for s in range(seeds):
        A = gen(n, *th, np.random.default_rng(root + 7919 * s))
        V.append(obs(A)); del A
    return np.array(V, float)


def rank_of(gen, n, th0, steps, seeds, root):
    V0 = sample(gen, n, th0, seeds, root)
    mu0 = V0.mean(axis=0)
    sd = V0.std(axis=0, ddof=1) / math.sqrt(seeds)
    floor = 1.0 / n / math.sqrt(seeds)      # пол: один отсчёт
    d = np.maximum(sd, floor)
    flat, shape = flat_pack(th0)
    cols = []
    for j, st in enumerate(steps):
        f2 = list(flat); f2[j] += st
        V1 = sample(gen, n, unpack(f2, shape), seeds, root + 555)
        cols.append((V1.mean(axis=0) - mu0) / d)
    J = np.column_stack(cols)
    # выбрасываем компоненты, где и дисперсия, и отклик ничтожны (0/0)
    keep = (np.abs(J).max(axis=1) > 0.5) | (sd > floor)
    J = J[keep]
    if J.size == 0:
        return np.zeros(len(steps)), 0, 0
    sv = np.linalg.svd(J, compute_uv=False)
    return sv, int((sv > 1.0).sum()), int(keep.sum())


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N, SD = 900, 8
    print(f'N = {N}, засевов {SD}, наблюдаемая: '
          f'{NDEP} полос глубины x {len(UED)-1} корзин + плотности '
          f'= {NDEP*(len(UED)-1)+NDEP} компонент')
    print()
    print('=' * 94)
    print('СЛОИСТОЕ СЕМЕЙСТВО: растёт ли различимых с числом слоёв L?')
    print(f"{'L':>4}{'параметров':>12}{'компонент':>11}{'РАЗЛИЧИМЫХ':>12}"
          f"{'первые сингулярные числа':>36}")
    for Lx in (3, 4, 6, 9, 12):
        parts = [1.0 / Lx] * Lx
        qs = [0.5] * (Lx - 1)
        steps = [0.04] * Lx + [0.06] * (Lx - 1)
        sv, r, nk = rank_of(gen_layers, N, (parts, qs), steps, SD, 5000 + Lx)
        s = ' '.join(f'{x:.1f}' for x in sv[:6])
        print(f'{Lx:>4}{2*Lx-1:>12}{nk:>11}{r:>12}{s:>36}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 94)
    print('КОНФОРМНАЯ МЕТРИКА: растёт ли с числом модов K?')
    print(f"{'K':>4}{'параметров':>12}{'компонент':>11}{'РАЗЛИЧИМЫХ':>12}"
          f"{'первые сингулярные числа':>36}")
    for K in (1, 2, 4, 8, 12, 20):
        sv, r, nk = rank_of(gen_conf, N, ([0.0] * K,), [0.05] * K,
                            SD, 6000 + K)
        s = ' '.join(f'{x:.1f}' for x in sv[:6])
        print(f'{K:>4}{K:>12}{nk:>11}{r:>12}{s:>36}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 94)
    print('''ЧТЕНИЕ. Смотреть НЕ на величину сингулярных числе, а на то,
растёт ли столбец «РАЗЛИЧИМЫХ» с числом параметров. Если у метрики он
растёт, а у слоистых упирается — размерность семейства у
пространства-времени больше, и оно беднее способами случиться, но
богаче способами быть.''')


if __name__ == '__main__':
    main()
