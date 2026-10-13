#!/usr/bin/env python3
"""СКЛЕЙКА как ЯЗЫК: точные ответы на малых блоках (быстрая версия).

Догадка: полоса N ~ 100…350, где ломаются все маршруты, есть размер
БЛОКА, на котором правило впервые выразимо, а целое собирается
склейкой «раньше-позже» с условием согласования. Это не мера на
конфигурациях, а ЯЗЫК — что разрешено приклеивать к чему.

Лаборатория d = 2: реализуемость есть РОВНО «порядковая размерность
<= 2», и проверяется точно.

  Q1. Ограничена ли склейка вообще.
  Q2. Как часто склейка законных блоков незаконна.
  Q3. Конечен ли список запретов — число МИНИМАЛЬНЫХ порядков
      размерности 3 по n. Если растёт, конечного списка нет, и
      локального правила в одном порядке не существует как теорема.
  Q4. Шаг роста Райдаута–Соркина: доля прошлых-подмножеств,
      сохраняющих законность. Держится — язык «растим»; обваливается —
      законность есть глобальный сговор.

Точный критерий: P = L1 ∩ L2 требует, чтобы L2 обращал КАЖДУЮ
несравнимую пару относительно L1, то есть L2 вынужден однозначно.
Значит достаточно перебрать линейные расширения L1 и проверить, что
вынужденный L2 транзитивен. Линейные расширения строятся рекурсивно
(не перебором n! перестановок), результат памятуется по каноническому
ключу матрицы.

КОНТРОЛЬ: цепь -> да, антицепь -> да, корона 3+3 -> нет (её
размерность ровно 3, классика).
"""
import itertools
import sys

import numpy as np

sys.setrecursionlimit(10000)
_memo = {}


def closure(R):
    n = len(R)
    C = R.copy()
    for k in range(n):
        C |= C[:, k][:, None] & C[k]
    return C


def lin_exts(R):
    n = len(R)
    preds = [set(np.flatnonzero(R[:, j]).tolist()) for j in range(n)]
    out, seq, used = [], [], set()

    def rec():
        if len(seq) == n:
            out.append(tuple(seq))
            return
        for x in range(n):
            if x not in used and preds[x] <= used:
                used.add(x); seq.append(x)
                rec()
                seq.pop(); used.discard(x)
    rec()
    return out


def dim_le_2(R):
    key = (len(R), R.astype(np.uint8).tobytes())
    if key in _memo:
        return _memo[key]
    n = len(R)
    inc = ~(R | R.T)
    np.fill_diagonal(inc, False)
    ans = False
    for seq in lin_exts(R):
        pos = np.empty(n, int)
        pos[list(seq)] = np.arange(n)
        # L2 ОБРАЩАЕТ несравнимые пары: i <_L2 j <=> j раньше i в L1.
        # (здесь был баг: сохранение порядка вместо обращения делало
        #  less2 равным самому L1, и проверка проходила всегда)
        rev = pos[None, :] < pos[:, None]
        less2 = R | (inc & rev)
        if (closure(less2.copy()) == less2).all() and \
           np.count_nonzero(less2) == n * (n - 1) // 2:
            ans = True
            break
    _memo[key] = ans
    return ans


def posets(n):
    """Неизоморфные порядки на n элементах, наращиванием по элементам."""
    cur = {np.zeros((1, 1), bool).tobytes(): np.zeros((1, 1), bool)}
    for m in range(2, n + 1):
        nxt = {}
        for R in cur.values():
            k = m - 1
            for bits in range(1 << k):
                D = np.array([(bits >> i) & 1 for i in range(k)], bool)
                if (R & D[None, :] & ~D[:, None]).any():
                    continue
                M = np.zeros((m, m), bool)
                M[:k, :k] = R
                M[:k, k] = D
                M = closure(M)
                best = None
                for p in itertools.permutations(range(m)):
                    q = np.array(p)
                    key = M[np.ix_(q, q)].astype(np.uint8).tobytes()
                    if best is None or key < best:
                        best = key
                nxt.setdefault(best, M)
        cur = nxt
    return list(cur.values())


def glue(RA, RB, I):
    a, b = len(RA), len(RB)
    R = np.zeros((a + b, a + b), bool)
    R[:a, :a] = RA; R[a:, a:] = RB; R[:a, a:] = I
    return closure(R)


def main():
    print(__doc__)
    ch = np.triu(np.ones((4, 4), bool), 1)
    crown = np.zeros((6, 6), bool)
    for i in range(3):
        for j in range(3):
            if i != j:
                crown[i, 3 + j] = True
    print('КОНТРОЛЬ')
    for nm, R, exp in (('цепь 4', ch, True),
                       ('антицепь 4', np.zeros((4, 4), bool), True),
                       ('корона 3+3', crown, False)):
        g = dim_le_2(R)
        print(f'  {nm:>12}: {g}  (ожидалось {exp})'
              f'{"" if g == exp else "   <-- ОШИБКА"}')

    print('\nQ1: ограничена ли склейка')
    A = B = np.zeros((3, 3), bool)
    I = np.array([[i != j for j in range(3)] for i in range(3)])
    C = glue(A, B, I)
    print(f'  A, B законны: {dim_le_2(A)}, {dim_le_2(B)}; '
          f'интерфейс замкнут: True; СКЛЕЙКА законна: {dim_le_2(C)}')
    print(f'  => склейка {"ОГРАНИЧЕНА" if not dim_le_2(C) else "свободна"}'
          f' (наименьший случай: 3+3)')

    print('\nQ2: доля незаконных склеек законных блоков')
    print(f"{'|A|':>4}{'|B|':>4}{'склеек':>10}{'незаконных':>12}{'доля':>8}")
    P = {k: [R for R in posets(k) if dim_le_2(R)] for k in (2, 3, 4)}
    for a, b in ((2, 3), (3, 3), (3, 4)):
        tot = bad = 0
        for RA in P[a]:
            for RB in P[b]:
                for bits in range(1 << (a * b)):
                    I = np.array([(bits >> t) & 1 for t in range(a * b)],
                                 bool).reshape(a, b)
                    if (RA @ I & ~I).any() or (I @ RB & ~I).any():
                        continue
                    tot += 1
                    if not dim_le_2(glue(RA, RB, I)):
                        bad += 1
        print(f'{a:>4}{b:>4}{tot:>10}{bad:>12}{bad/max(tot,1):>8.3f}',
              flush=True)

    print('\nQ3: число МИНИМАЛЬНЫХ порядков размерности 3')
    seq = []
    for n in (5, 6, 7):
        ps = posets(n)
        bad3 = [R for R in ps if not dim_le_2(R)]
        mini = 0
        for R in bad3:
            if all(dim_le_2(R[np.ix_([i for i in range(n) if i != d],
                                     [i for i in range(n) if i != d])])
                   for d in range(n)):
                mini += 1
        seq.append((n, len(ps), len(bad3), mini))
        print(f'  n={n}: порядков {len(ps)}, размерности>=3 {len(bad3)}, '
              f'МИНИМАЛЬНЫХ {mini}', flush=True)
    if len(seq) >= 2:
        g = [s[3] for s in seq]
        print(f'  минимальных по n: ' + ', '.join(
            f'n={s[0]}: {s[3]}' for s in seq))
        print('  => ' + ('список РАСТЁТ: конечным быть не может, значит '
                         'локального правила в одном порядке не '
                         'существует' if g[-1] > g[0] else
                         'на этом размахе не растёт — рано судить'))

    print('\nQ4: шаг роста — доля законных прошлых')
    print(f"{'|A|':>5}{'блоков':>8}{'шагов':>8}{'законных':>10}"
          f"{'доля':>8}{'мин. по блоку':>15}")
    for a in (2, 3, 4, 5, 6):
        PA = [R for R in posets(a) if dim_le_2(R)]
        tot = good = 0
        worst = 1.0
        for RA in PA:
            t = g = 0
            for bits in range(1 << a):
                D = np.array([(bits >> i) & 1 for i in range(a)], bool)
                if (RA & D[None, :] & ~D[:, None]).any():
                    continue
                M = np.zeros((a + 1, a + 1), bool)
                M[:a, :a] = RA; M[:a, a] = D
                t += 1
                if dim_le_2(closure(M)):
                    g += 1
            tot += t; good += g
            worst = min(worst, g / max(t, 1))
        print(f'{a:>5}{len(PA):>8}{tot:>8}{good:>10}'
              f'{good/max(tot,1):>8.3f}{worst:>15.3f}', flush=True)


if __name__ == '__main__':
    main()
