#!/usr/bin/env python3
"""Фаза из КИРАЛЬНОСТИ: даёт ли она показатель 1.

coher.py: фазы из порядка (доли общего прошлого) дают показатель
|A|^2 ~ |D|^1.998, то есть складываются когерентно и интерференции не
дают. Нужна величина с показателем 1, то есть НЕКОГЕРЕНТНАЯ между
соседями.

ИДЕЯ. Чётность ранжировки РАЗРЫВНА: даже если шкалы гладкие, порядок их
приращений перескакивает от события к событию. Значит знак +-1 по сути
случаен между соседями, и сумма пойдёт как случайное блуждание.

Киральность — архивный результат (ORDER-final.md §10): связь есть, если
ранжировка приращений чётная; нарушение 100% (отсечено 0 левых и 64485
правых), инвариантна к циклическому сдвигу шкал, не инвариантна к обмену
двух. В фазу её ни разу не ставили.

Проверяются амплитуды на ПАРЕ (v -> n), что и естественно для шага
роста: A = sum_{v in D} a(v, n).

    чётность (истинные шкалы)   a = exp(i*pi*parity) = +-1
    чётность (шкалы из порядка) то же, но шкалы = отношения счётов
    калибровочная фаза          a = exp(i*phi), phi = |общие предки| -
                                |общие потомки| (целое, архив: холономия
                                8.7 сигмы)
    гладкая из порядка          контроль, ожидается 2
    случайные                   контроль, ожидается 1
"""
import sys
import numpy as np

sys.path.insert(0, '/home/claude/code')
from weight import diamond, cone_rel

T, K = 2.0, 4
rs = np.random.default_rng(5301)
X = diamond(T, rs)
N = len(X)
R = cone_rel(X)
A = R.astype(np.float32)
sz = A.sum(1)

# истинные конусные шкалы
u = rs.normal(size=(K, 3))
u /= np.linalg.norm(u, axis=1, keepdims=True)
S_true = X[:, 0:1] - X[:, 1:] @ u.T

# шкалы из порядка: доли общего прошлого с k опорами
refs = np.argsort(-sz)[:K]
S_ord = np.zeros((N, K))
for m in range(K):
    S_ord[:, m] = (R & R[int(refs[m])][None, :]).sum(1) / np.maximum(sz, 1)

# гладкая фаза из порядка (как в coher.py)
ph_smooth = S_ord.sum(1)

# калибровочная фаза: общие предки минус общие потомки
CP = A @ A.T                                   # общие предки
CF = A.T @ A                                   # общие потомки
GAUGE = CP - CF

def parity(d):
    """Знак перестановки, сортирующей вектор приращений."""
    o = np.argsort(d, axis=-1)
    inv = 0
    k = d.shape[-1]
    for i in range(k):
        for j in range(i + 1, k):
            inv = inv + (o[..., i] > o[..., j])
    return 1 - 2 * (inv % 2)

print(f'N={N}, шкал {K}; показатель |A|^2 ~ |D|^alpha')
print('alpha = 1 интерференция работает, alpha = 2 фазы когерентны')
print()
print(f"{'|D|':>6}{'чёт.истин.':>13}{'чёт.порядк.':>13}"
      f"{'калибр.фаза':>13}{'гладкая':>11}{'случайные':>11}")
sizes, cols = [], {k: [] for k in
                   ('чи', 'чп', 'кф', 'гл', 'сл')}
rnd = np.exp(2j * np.pi * rs.random(N))
for target in (4, 8, 16, 32, 64, 128):
    acc = {k: [] for k in cols}
    for _ in range(300):
        n = int(rs.integers(0, N))
        p = np.flatnonzero(R[n])
        if len(p) < target:
            continue
        D = rs.choice(p, target, replace=False)
        dt = S_true[n] - S_true[D]
        do = S_ord[n] - S_ord[D]
        acc['чи'].append(abs(parity(dt).sum()) ** 2)
        acc['чп'].append(abs(parity(do).sum()) ** 2)
        acc['кф'].append(abs(np.exp(1j * GAUGE[n, D]).sum()) ** 2)
        acc['гл'].append(abs(np.exp(2j * np.pi * ph_smooth[D]).sum()) ** 2)
        acc['сл'].append(abs(rnd[D].sum()) ** 2)
    if not acc['чи']:
        continue
    sizes.append(target)
    for k in cols:
        cols[k].append(np.mean(acc[k]))
    print(f'{target:>6}{np.mean(acc["чи"]):>13.1f}'
          f'{np.mean(acc["чп"]):>13.1f}{np.mean(acc["кф"]):>13.1f}'
          f'{np.mean(acc["гл"]):>11.1f}{np.mean(acc["сл"]):>11.1f}')

s = np.log(sizes)
print()
nm = {'чи': 'чётность (истинные шкалы)', 'чп': 'чётность (из порядка)',
      'кф': 'калибровочная фаза', 'гл': 'гладкая из порядка',
      'сл': 'случайные'}
for k in ('чи', 'чп', 'кф', 'гл', 'сл'):
    a = np.polyfit(s, np.log(np.maximum(cols[k], 1e-9)), 1)[0]
    print(f'  {nm[k]:>26}: показатель {a:.3f}')
