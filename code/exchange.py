import numpy as np
from collections import deque

def can_swap(k, m=7):
    """Две неразличимые копии в k-мерном кубе фаз (сетка m^k).
    Конфигурация — упорядоченная пара разных клеток. Ход: одна копия
    сдвигается в соседнюю клетку, не совпадая со второй.
    Вопрос: связаны ли (a,b) и (b,a)?"""
    cells = np.array(np.unravel_index(np.arange(m**k), (m,)*k)).T   # все клетки
    idx = {tuple(c): i for i, c in enumerate(cells)}
    nbr = [[] for _ in range(len(cells))]
    for i, c in enumerate(cells):
        for d in range(k):
            for s in (-1, 1):
                q = c.copy(); q[d] += s
                if 0 <= q[d] < m: nbr[i].append(idx[tuple(q)])
    n = len(cells)
    start = (0, n - 1)                       # две противоположные клетки
    goal = (n - 1, 0)                        # они же, переставленные
    seen = {start}; dq = deque([start])
    while dq:
        a, b = dq.popleft()
        if (a, b) == goal: return True, len(seen)
        for x in nbr[a]:
            if x != b and (x, b) not in seen:
                seen.add((x, b)); dq.append((x, b))
        for y in nbr[b]:
            if y != a and (a, y) not in seen:
                seen.add((a, y)); dq.append((a, y))
    return False, len(seen)
