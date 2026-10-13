#!/usr/bin/env python3
"""ПОДСТАНОВОЧНЫЕ ФРАКТАЛЫ: ищем затравку, дающую профиль засева.

Фрактал задаётся ПРАВИЛОМ ПОСТРОЕНИЯ, а не штрафом - то есть ровно тем
способом, который по `tri-doc.md` единственно и может работать (локальный
вес бессилен против энтропии n^2, но кинематическое ограничение её
снимает).

Построение: затравка B на b элементах, каждый элемент заменяется копией
B, повторяя k раз. В матрицах одна строка:
    R_{k+1} = kron(R_B, единицы) | kron(I_b, R_k)
то есть (p,r) < (p',r') тогда и только тогда, когда p < p' в B, либо
p = p' и r < r'.

ДВА ПРЕДСКАЗАНИЯ ДО ЗАМЕРА (так надёжнее):

(1) ТОЧНОЕ. Для числа связанных пар C_{k+1} = C_B b^{2k} + b C_k, значит
    при u_k = C_k/b^{2k} имеем u_{k+1} = C_B/b^2 + u_k/b, неподвижная
    точка u* = C_B/(b(b-1)) = r_B/2. То есть ДОЛЯ СВЯЗАННЫХ У ФРАКТАЛА
    РАВНА ДОЛЕ СВЯЗАННЫХ У ЗАТРАВКИ, ровно. Это встроенный контроль.

(2) КАЧЕСТВЕННОЕ. Интервал между (p,r) и (p',r') равен
    |I_B(p,p')| b^k + |вверх от r| + |вниз от r'|. Значит если в
    затравке между p и p' есть хоть один элемент, интервал сразу ОГРОМЕН
    (>= b^k), а малые интервалы дают только СВЯЗИ затравки. Спектр
    размеров интервалов дискретен и с провалами (кучкуется у степеней b),
    тогда как у засева он гладкий - это и есть закон Gamma(m+1/2)/m!.
    Предсказание: фрактал профиль НЕ пройдёт, и именно по этой причине.

Поиск идёт по ВСЕМ затравкам на 4, 5 и 6 элементах (16 + 63 + 318), то
есть по всему семейству правил такого вида при малом b.
"""
import math

import numpy as np

from compose2 import posets
from sectors import sprinkle_d
from ratios import nm

MS = tuple(range(1, 9))


def frac_poset(RB, k):
    b = RB.shape[0]
    R = RB.copy()
    for _ in range(k - 1):
        m = R.shape[0]
        R = (np.kron(RB, np.ones((m, m), bool))
             | np.kron(np.eye(b, dtype=bool), R))
    return R


def prof(Rc):
    v, _ = nm(Rc)
    return v[1:] / max(v[0], 1.0)


def region(n, rs, rep):
    A = np.array([prof(sprinkle_d(n, 4, rs)) for _ in range(rep)])
    mu, sd = A.mean(axis=0), A.std(axis=0, ddof=1)
    chi = lambda X: (((X - mu) / sd) ** 2).sum(axis=-1)
    return mu, sd, chi, float(np.percentile(chi(A), 95))


def main():
    print(__doc__)
    rs = np.random.default_rng(8164965)

    print('КОНТРОЛЬ предсказания (1): доля связанных фрактала = r затравки')
    for RB in list(posets(5))[:4] + list(posets(4))[:2]:
        b = RB.shape[0]
        rb = RB.sum() / (b * (b - 1) / 2)
        line = []
        for k in (2, 3, 4):
            R = frac_poset(RB, k)
            n = R.shape[0]
            line.append(R.sum() / (n * (n - 1) / 2))
        print(f'  b={b} r_B={rb:.4f}  ->  фрактал: '
              + ' '.join(f'{v:.4f}' for v in line)
              + f'   |откл| при k=4: {abs(line[-1] - rb):.2e}')

    print('\nПОИСК по всем затравкам (b, k, N):')
    for b, k, rep in ((4, 5, 30), (5, 4, 30), (6, 3, 40)):
        n = b ** k
        mu, sd, chi, thr = region(n, rs, rep)
        best = None
        cnt = 0
        for RB in posets(b):
            if RB.shape[0] != b:
                continue
            cnt += 1
            R = frac_poset(RB, k)
            c = float(chi(prof(R)))
            rb = RB.sum() / (b * (b - 1) / 2)
            if best is None or c < best[0]:
                best = (c, rb, RB.copy())
        print(f'  b={b} k={k} N={n}: затравок {cnt}, порог хи2={thr:.1f}, '
              f'НАИЛУЧШАЯ хи2={best[0]:.1f} (r_B={best[1]:.3f})')

    print('\nСПЕКТР размеров интервалов: фрактал против засева (N=625)')
    RB = max((R for R in posets(5) if R.shape[0] == 5),
             key=lambda R: abs(R.sum() / 10 - 0.1) * -1)
    R = frac_poset(RB, 4)
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    szf = Iv[R]
    S = sprinkle_d(625, 4, rs)
    F2 = S.astype(np.float32)
    Iv2 = np.rint(F2 @ F2).astype(np.int32)
    szs = Iv2[S]
    for lab, sz in (('фрактал', szf), ('засев ', szs)):
        h = [int((sz == m).sum()) for m in range(9)]
        print(f'  {lab}: |I|=0..8 -> ' + ' '.join(f'{v:>6}' for v in h)
              + f'   |I|>8: {int((sz > 8).sum())},  '
              f'различных значений |I|: {len(np.unique(sz))}')


if __name__ == '__main__':
    main()
