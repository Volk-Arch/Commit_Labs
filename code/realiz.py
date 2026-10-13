#!/usr/bin/env python3
"""ОПРЕДЕЛИМ ЛИ ЧЕТЫРЁХМЕРНЫЙ КЛАСС ЛОКАЛЬНЫМИ ЗАПРЕТАМИ.

Правило вида «не даёт появляться лишнему» = конечный список
ЗАПРЕЩЁННЫХ подпорядков. Он локален автоматически: любая конфигурация
впервые появляется ровно при рождении своего максимального элемента,
значит легальность проверяется в момент рождения.

Но запрещать можно лишь то, чего в пространстве-времени НЕ бывает.
Поэтому решающий вопрос: какие малые порядки НЕреализуемы как
причинный порядок точек в d-мерном Минковском.

ИЗВЕСТНОЕ, и оно же КОНТРОЛЬ: при d = 2 причинные множества суть в
точности порядки порядковой размерности <= 2 (светоконусные координаты
u = t+x, v = t-x дают две линейные оси). Наименьший порядок размерности
3 — «корона» на шести элементах: три минимальных, три максимальных,
каждый минимальный под двумя максимальными. Значит при d = 2 поиск
ОБЯЗАН её не найти, а при d >= 3 обязан найти. Если контроль не
сработает — плох поиск, а не ответ.

  python3 realiz.py
"""
import itertools
import math

import numpy as np

COUNTS = {1: 1, 2: 2, 3: 5, 4: 16, 5: 63, 6: 318}


def closure(R):
    n = R.shape[0]
    C = R.copy()
    while True:
        nxt = C | ((C.astype(np.int8) @ C.astype(np.int8)) > 0)
        if (nxt == C).all():
            return C
        C = nxt


def canon(R):
    """Канонический вид: минимум по перестановкам."""
    n = R.shape[0]
    best = None
    for p in itertools.permutations(range(n)):
        q = np.array(p)
        key = R[np.ix_(q, q)].astype(np.uint8).tobytes()
        if best is None or key < best:
            best = key
    return best


def all_posets(n):
    """Все неизоморфные порядки на n элементах."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    seen = {}
    for m in range(1 << len(pairs)):
        R = np.zeros((n, n), dtype=bool)
        for b, (i, j) in enumerate(pairs):
            if (m >> b) & 1:
                R[i, j] = True
        C = closure(R)
        seen.setdefault(canon(C), C)
    return seen


def crown_key(n=6):
    """«Корона» S_3^0: 0,1,2 минимальные; 3,4,5 максимальные;
    i < 3+j для j != i."""
    R = np.zeros((6, 6), dtype=bool)
    for i in range(3):
        for j in range(3):
            if i != j:
                R[i, 3 + j] = True
    return canon(closure(R)), R


def causal(P):
    """Причинный порядок набора точек (t, x...)."""
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt ** 2 > dx2)


def sample_keys(n, d, tries, rs, batch=4000):
    """Какие классы встречаются у случайных точек в Минковском."""
    found = set()
    done = 0
    while done < tries:
        k = min(batch, tries - done)
        pts = rs.random((k, n, d))
        pts[:, :, 0] *= 2.0        # побольше времени, чтобы были связи
        for i in range(k):
            found.add(canon(causal(pts[i])))
        done += k
    return found


def anneal(n, d, target, rs, iters=30000, restarts=12):
    """Направленный поиск: найти точки, дающие ИМЕННО target."""
    tgt = target
    for _ in range(restarts):
        P = rs.random((n, d))
        P[:, 0] *= 2.0
        if canon(causal(P)) == tgt:
            return P
        T0 = 0.35
        cur = P.copy()
        curbad = np.inf
        for it in range(iters):
            T = T0 * (1 - it / iters) + 1e-3
            Q = cur.copy()
            i = rs.integers(n)
            Q[i] += rs.normal(scale=T, size=d)
            k = canon(causal(Q))
            if k == tgt:
                return Q
            # штраф: число несовпавших пар с лучшим выравниванием target
            bad = hamming_min(causal(Q), tgt, n)
            if bad <= curbad or rs.random() < math.exp(-(bad - curbad) / T):
                cur, curbad = Q, bad
    return None


def hamming_min(R, tgtkey, n):
    T = np.frombuffer(tgtkey, dtype=np.uint8).reshape(n, n).astype(bool)
    best = n * n
    for p in itertools.permutations(range(n)):
        q = np.array(p)
        v = int((R[np.ix_(q, q)] != T).sum())
        if v < best:
            best = v
    return best


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(10092026)
    print('КОНТРОЛЬ: число неизоморфных порядков')
    pos = {}
    for n in (3, 4, 5, 6):
        pos[n] = all_posets(n)
        ok = 'ВЕРНО' if len(pos[n]) == COUNTS[n] else 'НЕВЕРНО'
        print(f'  n = {n}: найдено {len(pos[n]):>4}, '
              f'должно быть {COUNTS[n]:>4}  {ok}', flush=True)
    ck, cR = crown_key()
    print(f'  «корона» среди порядков на 6 элементах: '
          f'{"да" if ck in pos[6] else "НЕТ"}')
    print()
    print('РЕАЛИЗУЕМОСТЬ: сколько классов встречается у случайных точек')
    print(f"{'n':>4}{'d':>4}{'классов всего':>15}{'найдено случайно':>18}"
          f"{'не найдено':>12}")
    missing = {}
    for n in (4, 5, 6):
        for d in (2, 3, 4):
            tries = {4: 400000, 5: 250000, 6: 120000}[n]
            f = sample_keys(n, d, tries, rs)
            f &= set(pos[n])
            miss = set(pos[n]) - f
            missing[(n, d)] = miss
            print(f'{n:>4}{d:>4}{len(pos[n]):>15}{len(f):>18}'
                  f'{len(miss):>12}', flush=True)
    print()
    print('КОНТРОЛЬ: корона при d = 2 и d >= 3')
    for d in (2, 3, 4):
        inmiss = ck in missing[(6, d)]
        print(f'  d = {d}: корона {"НЕ найдена" if inmiss else "найдена"} '
              f'случайным поиском')
    print()
    print('НАПРАВЛЕННЫЙ ПОИСК по ненайденным (это и решает вопрос)')
    for n in (4, 5, 6):
        for d in (2, 3, 4):
            miss = missing[(n, d)]
            if not miss:
                print(f'  n={n}, d={d}: ненайденных нет')
                continue
            still = []
            for key in sorted(miss):
                P = anneal(n, d, key, rs)
                if P is None:
                    still.append(key)
            print(f'  n={n}, d={d}: было ненайдено {len(miss)}, '
                  f'после поиска осталось {len(still)}'
                  f'{"  <- КОРОНА среди них" if ck in still else ""}',
                  flush=True)
            missing[(n, d)] = set(still)
    print()
    print('ИТОГ: нереализуемые классы (кандидаты в запреты)')
    for (n, d), miss in sorted(missing.items()):
        print(f'  n={n}, d={d}: {len(miss)}')


if __name__ == '__main__':
    main()
