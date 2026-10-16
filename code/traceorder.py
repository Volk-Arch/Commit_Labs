#!/usr/bin/env python3
"""ПОРЯДОК ПО ПОЛНОМУ СЛЕДУ (1+1, плоский золотой мир gold2).

Событие: (T, x | T', x'), тень (T', x') в круге радиуса r.
Старое правило: A ≺ B ⟺ ΔT > |Δx| (только наш мир).
Новое правило: A ≺ B ⟺ ΔT > 0 и ΔT² − Δx² − ΔT'² − Δx'² > 0 — след разности, вместе с тенью.
(В масштабе решётки это Tr(τΔt² − Δx²) — целое число.)

Что смотрим: (а) сколько пар теряется, связи; (б) самые длинные цепочки: держат ли они
своё место в окне (внутренний вектор), или бегают по окну как попало; (в) длины цепочек.
"""
import sys, math, time, numpy as np
sys.path.insert(0, '/home/claude/code')
from bentcut import zt_range, ST, SI

def world(rho, L, rs):
    r = math.sqrt(5 * rho / math.pi); sh = rs.uniform(0, 300, 2); so = rs.uniform(-1, 1, 2)
    t, tp = zt_range((sh[0] - L) / ST, (sh[0] + L) / ST, (so[0] - r) / SI, (so[0] + r) / SI); T, Tq = t * ST - sh[0], tp * SI - so[0]
    c, cp = zt_range(sh[1] - L, sh[1] + L, so[1] - r, so[1] + r); c, cp = c - sh[1], cp - so[1]
    acc = Tq[:, None] ** 2 + cp[None, :] ** 2 <= r * r
    i, j = np.nonzero(acc)
    P = np.stack([T[i], c[j], Tq[i], cp[j]], 1)
    P = P[np.abs(P[:, 0]) + np.abs(P[:, 1]) <= L]               # причинный ромб
    return P[np.argsort(P[:, 0])], r

def order(P, full):
    d = P[None, :, :] - P[:, None, :]                            # d[i,j] = P[j] − P[i]
    s = d[..., 0] ** 2 - d[..., 1] ** 2
    if full: s = s - d[..., 2] ** 2 - d[..., 3] ** 2
    return (d[..., 0] > 0) & (s > 0)

def longest(R, a, b):
    """самая длинная цепочка a -> b (индексы по времени), возвращает список событий."""
    n = len(R); idx = np.nonzero(R[a] & R[:, b])[0]; idx = np.concatenate([[a], idx, [b]])
    sub = R[np.ix_(idx, idx)]; m = len(idx); dist = np.full(m, -10 ** 9); dist[0] = 0; pred = np.full(m, -1)
    for k in range(1, m):
        p = np.nonzero(sub[:k, k])[0]
        if len(p): j = p[np.argmax(dist[p])]; dist[k] = dist[j] + 1; pred[k] = j
    path = [m - 1]
    while path[-1] != 0: path.append(pred[path[-1]])
    return idx[np.array(path[::-1])], dist[m - 1]

if __name__ == '__main__':
    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0; L = float(sys.argv[2]) if len(sys.argv) > 2 else 22.0
    NW = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rs = np.random.default_rng(3); t0 = time.time()
    stats = {False: [], True: []}
    for w in range(NW):
        P, r = world(rho, L, rs); n = len(P)
        Rp, Rf = order(P, False), order(P, True)
        lost = 1 - Rf.sum() / Rp.sum()
        # связи (непосредственные): i ≺ j и нет k между
        def links(R):
            Ri = R.astype(np.int32); return (R & ((Ri @ Ri) == 0)).sum(1).mean()
        print(f'мир {w}: событий {n}, окно r = {r:.2f}; пар потеряно новым правилом {lost:.4f}; связей на событие: старое {links(Rp):.2f}, новое {links(Rf):.2f}  [{time.time() - t0:.0f} с]')
        # цепочки между событиями у низа и верха ромба
        bot = np.nonzero(P[:, 0] < -L + 3)[0]; top = np.nonzero(P[:, 0] > L - 3)[0]
        for _ in range(12):
            a, b = rs.choice(bot), rs.choice(top)
            if not (Rp[a, b] and Rf[a, b]): continue
            for full, R in ((False, Rp), (True, Rf)):
                ch, ln = longest(R, a, b)
                sh = P[ch][:, 2:]                                    # место в окне вдоль цепочки
                lin = P[a, 2:] + (P[ch, 0] - P[a, 0])[:, None] / (P[b, 0] - P[a, 0]) * (P[b, 2:] - P[a, 2:])[None, :]
                dev = np.sqrt(((sh - lin) ** 2).sum(1)).mean()        # отклонение от прямой между концами
                step = np.sqrt((np.diff(sh, axis=0) ** 2).sum(1)).mean()
                stats[full].append((ln, dev, step, len(ch)))
    for full in (False, True):
        A = np.array(stats[full])
        print(f'{"НОВОЕ (след)" if full else "СТАРОЕ (наш мир)"}: длина цепочки {A[:, 0].mean():.1f}; место в окне: отклонение от прямой {A[:, 1].mean():.3f}, шаг между соседями {A[:, 2].mean():.3f}   (случайные точки окна: отклонение ≈ {0.9 * r:.2f}, шаг ≈ {0.9 * r:.2f})  [{len(A)} цепочек]')
