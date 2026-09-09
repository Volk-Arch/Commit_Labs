import numpy as np

def sprinkle(N, d, rs, T=1.0):
    pts = []
    while len(pts) < N:
        m = 8*(N-len(pts))+300
        t = rs.uniform(0, T, m); x = rs.uniform(-T/2, T/2, (m, d-1))
        r = np.linalg.norm(x, axis=1); ok = (r < t) & (r < T-t)
        pts.extend(np.column_stack([t[ok], x[ok]]))
    P = np.array(pts[:N]); return P[np.argsort(P[:, 0])]

def order_of(P):
    t = P[:, 0]; x = P[:, 1:]
    dt = t[None, :]-t[:, None]
    dx2 = ((x[None, :, :]-x[:, None, :])**2).sum(-1)
    return ((dt > 0) & (dt**2 > dx2))      # M[i,j] = True  <=> i предшествует j

def largest_standard_example(M, tries=300, rs=None, cap=12):
    """Наибольшее n, при котором в порядок вложен S_n.
    Даёт СТРОГУЮ нижнюю оценку порядковой размерности: dim >= n.

    S_n: пары (a_i, b_i) несравнимы, но a_i < b_j для всех i != j.
    Ищем максимальную клику в графе совместимости пар.
    """
    r = rs if rs is not None else np.random.default_rng(0)
    N = len(M)
    inc = ~M & ~M.T
    np.fill_diagonal(inc, False)
    pairs = np.argwhere(np.triu(inc))          # несравнимые пары (a,b), a<b по индексу
    if len(pairs) < 2: return 1
    if len(pairs) > 2500:
        pairs = pairs[r.choice(len(pairs), 2500, replace=False)]
    P = len(pairs)
    A = pairs[:, 0]; B = pairs[:, 1]
    # совместимость: (a1,b1) и (a2,b2) образуют кусок S_2, если a1<b2 и a2<b1
    C1 = M[np.ix_(A, B)]                       # C1[p,q] = a_p < b_q
    comp = C1 & C1.T
    # плюс все четыре элемента различны
    for arr in (A, B):
        for arr2 in (A, B):
            comp &= (arr[:, None] != arr2[None, :])
    best = 1
    for _ in range(tries):
        order = r.permutation(P)
        clique = []
        for v in order:
            if all(comp[v, u] for u in clique):
                clique.append(v)
                if len(clique) >= cap: break
        best = max(best, len(clique))
    return best
