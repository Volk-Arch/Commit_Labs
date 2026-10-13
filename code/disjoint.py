#!/usr/bin/env python3
"""Петля ПРИ НАЛИЧИИ ВЕЩЕСТВА: конус — неподвижная точка или нет.

ruler.py: линейка требует вещества. Естественная цепь вакуумного засева
даёт отношение разбросов 0.408 при нужных 1.370 и корреляцию с
настоящей проекцией не выше -0.54. Подсаженная плотная линия проходит
порог около 120 событий (отношение 1.697).

Значит вопрос ставится условно: ЕСЛИ вещество уже есть, воспроизводит
ли конструкция себя? Опоры — m плотных мировых линий, шкалы — попарные
знаковые разности, k = m-1 независимых направлений.

Это не вывод вещества (его подсаживаем руками), а проверка, замыкается
ли петля при его наличии. Если да — зазор сужается до одного вопроса:
откуда первое вещество.
"""
import sys
import numpy as np

sys.path.insert(0, '/home/claude/code')
from core import frac_pairs, mm_dim, height, spectral, transitivity_violations
from fixpoint import diamond, cone_order, box_order, rel_of_int
from coneloop import heights_of

N0, T, M, NW = 1500, 2.0, 140, 8   # 8 линий -> 4 НЕПЕРЕСЕКАЮЩИЕСЯ пары
ITERS = 7

def Wof(R, ch):
    rel = R[:, ch] | R[ch, :].T
    return (~rel).sum(1).astype(float)

def scales(R, chains):
    """Шкалы по НЕПЕРЕСЕКАЮЩИМСЯ парам опор.

    Прежняя версия брала W_i - W_0 от ОДНОЙ общей опоры — это звезда, а
    не независимые направления: общая W_0 коррелирует шкалы между собой
    по построению (как неортогональный базис). Непересекающиеся пары
    дают независимые направления, как ортогональные u у конуса.
    """
    h = heights_of(R).astype(float)
    Ws = [Wof(R, c) for c in chains]
    V = np.stack([h - (Ws[2 * i + 1] - Ws[2 * i]) / 2.0
                  for i in range(len(Ws) // 2)], 1)
    return V

rs = np.random.default_rng(4401)
X0 = diamond(N0, T, rs)
parts, chains, n = [X0], [], len(X0)
for s in range(NW):
    c = rs.uniform(-0.40, 0.40, 3)
    Y = np.empty((M, 4))
    Y[:, 0] = np.sort(rs.uniform(0.25, 1.75, M))
    Y[:, 1:] = c + 0.01 * rs.normal(size=(M, 3))
    parts.append(Y); chains.append(np.arange(n, n + M)); n += M
X = np.vstack(parts)
N = len(X)
print(f'N={N} ({N0} фон + {NW} линий по {M}), шкал {NW//2} '
      f'(непересекающиеся пары)')
V0 = scales(starts_probe := None, None) if False else None
print('шкала v = h - (W_i - W_0)/2; вещество подсажено руками')
print('ЭТАЛОН конуса: доля ~0.10, разм. 4.00, спектр > 0')
print()

Rc = cone_order(X)
Vd = scales(Rc, chains)
iu = np.triu_indices(Vd.shape[1], 1)
print(f'корреляция МЕЖДУ шкалами на конусе: '
      f'{np.corrcoef(Vd.T)[iu].mean():+.4f}   (звезда давала +0.91, '
      f'конус +0.24)')
print()
starts = {'конус': Rc, 'ящик': box_order(N, rs),
          'случайный': rel_of_int(rs.random((N, 2)))}
fin = {}
for nm, R in starts.items():
    print(f'НАЧАЛО: {nm}')
    print(f"{'итерация':>12}{'доля пар':>10}{'разм.':>9}{'высота':>8}"
          f"{'спектр':>10}{'наруш':>8}")
    def rep(R, l):
        f = frac_pairs(R)
        print(f'{l:>12}{f:>10.4f}{mm_dim(f):>9.2f}{height(R)[0]:>8.0f}'
              f'{spectral(R):>+10.4f}{transitivity_violations(R):>8}',
              flush=True)
        return f
    rep(R, '0')
    prev = None
    for it in range(1, ITERS + 1):
        V = scales(R, chains) + 1e-9 * rs.random((N, NW // 2))
        Rn = rel_of_int(V)
        same = float((Rn == R).mean())
        R = Rn
        f = rep(R, str(it))
        if prev is not None and abs(f - prev) < 1e-4 and same > 0.9999:
            print(f'   неподвижная точка на итерации {it}')
            break
        prev = f
    fin[nm] = R
    print()

ns = list(fin)
print('СОШЛИСЬ ЛИ РАЗНЫЕ НАЧАЛА В ОДНО')
for i in range(len(ns)):
    for j in range(i + 1, len(ns)):
        A, B = fin[ns[i]], fin[ns[j]]
        print(f'  {ns[i]:>10} против {ns[j]:<10} совпадение '
              f'{float((A == B).mean()):.4f}   доли '
              f'{frac_pairs(A):.4f} / {frac_pairs(B):.4f}')
