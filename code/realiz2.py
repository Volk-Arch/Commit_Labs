#!/usr/bin/env python3
"""ОПРЕДЕЛИМ ЛИ ЧЕТЫРЁХМЕРНЫЙ КЛАСС ЛОКАЛЬНЫМИ ЗАПРЕТАМИ. Версия 2.

Правило «не даёт появляться лишнему» = конечный список ЗАПРЕЩЁННЫХ
подпорядков, и он локален автоматически: конфигурация впервые
появляется при рождении своего максимального элемента. Но запрещать
можно лишь то, чего в пространстве-времени НЕ бывает, поэтому вопрос:
какие малые порядки НЕреализуемы в d-мерном Минковском.

КОНТРОЛЬ, он же известный факт: при d = 2 причинные множества суть в
точности порядки порядковой размерности <= 2 (светоконусные координаты
дают две линейные оси). Наименьший порядок размерности 3 — «корона» на
шести элементах. При d = 2 она обязана быть НЕреализуема, при d >= 3 —
реализуема.

Ускорения против версии 1: канонизация векторизована по батчам;
направленный поиск идёт по ФИКСИРОВАННОЙ разметке цели (порядок
реализуем тогда и только тогда, когда реализуема любая его разметка),
поэтому перестановки в цикле не нужны.

  python3 realiz2.py
"""
import itertools
import math

import numpy as np

COUNTS = {3: 5, 4: 16, 5: 63, 6: 318}


def perms_of(n):
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


def canon_batch(R, P, W):
    """R: (B,n,n) bool -> (B,) int64 канонический ключ."""
    B, n, _ = R.shape
    Rp = R[:, P[:, :, None], P[:, None, :]]          # (B,Pn,n,n)
    keys = Rp.reshape(B, P.shape[0], n * n).astype(np.float64) @ W
    return keys.min(axis=1).astype(np.int64)


def closure_batch(R):
    C = R.copy()
    for _ in range(R.shape[1]):
        nxt = C | (np.einsum('bij,bjk->bik', C.astype(np.int8),
                             C.astype(np.int8)) > 0)
        if (nxt == C).all():
            break
        C = nxt
    return C


def all_posets(n):
    P, W = perms_of(n), 2.0 ** np.arange(n * n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    out = {}
    tot = 1 << len(pairs)
    for s in range(0, tot, 512):
        e = min(s + 512, tot)
        B = e - s
        R = np.zeros((B, n, n), dtype=bool)
        for b, (i, j) in enumerate(pairs):
            bit = ((np.arange(s, e) >> b) & 1).astype(bool)
            R[bit, i, j] = True
        C = closure_batch(R)
        k = canon_batch(C, P, W)
        for idx in range(B):
            out.setdefault(int(k[idx]), C[idx].copy())
    return out


def crown():
    R = np.zeros((6, 6), dtype=bool)
    for i in range(3):
        for j in range(3):
            if i != j:
                R[i, 3 + j] = True
    return R


def causal_one(p):
    dt = p[:, 0][None, :] - p[:, 0][:, None]
    dx2 = ((p[:, 1:][None, :, :] - p[:, 1:][:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt * dt > dx2)


def causal_batch(pts):
    dt = pts[:, None, :, 0] - pts[:, :, None, 0]
    dx = pts[:, None, :, 1:] - pts[:, :, None, 1:]
    return (dt > 0) & (dt * dt > (dx ** 2).sum(-1))


def found_keys(n, d, tries, rs, P, W, batch=256):
    got = set()
    done = 0
    while done < tries:
        k = min(batch, tries - done)
        pts = rs.random((k, n, d))
        pts[:, :, 0] *= 2.0
        R = causal_batch(pts)
        for v in canon_batch(R, P, W):
            got.add(int(v))
        done += k
    return got


def search(T, d, rs, iters=4000, restarts=8):
    """Найти точки в d-мерном Минковском с ТОЧНО таким порядком T."""
    n = T.shape[0]
    for _ in range(restarts):
        p = rs.random((n, d))
        p[:, 0] *= 2.0
        bad = int((causal_one(p) != T).sum())
        if bad == 0:
            return p
        for it in range(iters):
            T_ = 0.4 * (1 - it / iters) + 0.01
            q = p.copy()
            q[rs.integers(n)] += rs.normal(scale=T_, size=d)
            nb = int((causal_one(q) != T).sum())
            if nb < bad or (nb == bad and rs.random() < 0.5):
                p, bad = q, nb
                if bad == 0:
                    return p
    return None


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(10092026)
    print('КОНТРОЛЬ: число неизоморфных порядков')
    pos = {}
    for n in (3, 4, 5, 6):
        pos[n] = all_posets(n)
        print(f'  n = {n}: {len(pos[n]):>4} (должно {COUNTS[n]:>4})  '
              f'{"ВЕРНО" if len(pos[n])==COUNTS[n] else "НЕВЕРНО"}',
              flush=True)
    P6, W6 = perms_of(6), 2.0 ** np.arange(36)
    ckey = int(canon_batch(crown()[None], P6, W6)[0])
    print(f'  корона есть в списке: {"да" if ckey in pos[6] else "НЕТ"}')
    print()
    print('РЕАЛИЗУЕМОСТЬ')
    print(f"{'n':>3}{'d':>3}{'всего':>7}{'случайно':>10}"
          f"{'после поиска':>14}{'НЕреализуемо':>14}")
    res = {}
    for n in (4, 5, 6):
        Pn, Wn = perms_of(n), 2.0 ** np.arange(n * n)
        for d in (2, 3, 4):
            tries = {4: 60000, 5: 40000, 6: 30000}[n]
            got = found_keys(n, d, tries, rs, Pn, Wn) & set(pos[n])
            miss = sorted(set(pos[n]) - got)
            still = []
            for key in miss:
                if search(pos[n][key], d, rs) is None:
                    still.append(key)
            res[(n, d)] = still
            print(f'{n:>3}{d:>3}{len(pos[n]):>7}{len(got):>10}'
                  f'{len(pos[n])-len(still):>14}{len(still):>14}',
                  flush=True)
    print()
    print('КОНТРОЛЬ КОРОНЫ (решает, можно ли верить поиску)')
    for d in (2, 3, 4):
        bad = ckey in res[(6, d)]
        want = 'НЕреализуема' if d == 2 else 'реализуема'
        got = 'НЕреализуема' if bad else 'реализуема'
        print(f'  d = {d}: ожидалось {want:>13}, получено {got:>13}  '
              f'{"ВЕРНО" if (d == 2) == bad else "ПРОВАЛ"}')
    print()
    print('ИТОГ')
    for d in (2, 3, 4):
        tot = sum(len(res[(n, d)]) for n in (4, 5, 6))
        print(f'  d = {d}: нереализуемых классов на 4-6 элементах: {tot}')


if __name__ == '__main__':
    main()
