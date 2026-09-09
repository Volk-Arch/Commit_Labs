"""Голономия спинора по петле = зацепление петли с осями поворотов на π. См. docs/spin-doc.md §16.

Предсказание. Направленный подъём (обратный переход через стенку x_a=x_b = обратный
элемент) даёт по замкнутой петле знак
    (−1)^(число страт {x_i=x_j, x_l=x_m}, протыкающих диск петли),
а тройные страты {x_i=x_j=x_l} не влияют. Страты {x_i=x_j, x_l=x_m} — неподвижные
множества поворотов на π (двойных транспозиций), у которых знак подъёма вынужден
(spin_forced.py). Конвенция кода (spin_loop в simulate.py) даёт (−1)^(смен/2) × это.
Проверка на случайных петлях где угодно в кубе и на обменах двух копий.

Запуск: python spin_holonomy.py
"""
import itertools
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from simulate import clifford, transposition  # noqa: E402


def refine(P, closed=True):
    """Делим шаги пополам, пока каждая смена ранжировки не станет одной транспозицией."""
    n = len(P)
    pairs = [(P[t], P[(t + 1) % n]) for t in range(n if closed else n - 1)]
    out = [P[0]]
    stack = list(reversed(pairs))
    while stack:
        a, b = stack.pop()
        if (np.argsort(a) != np.argsort(b)).sum() > 2:
            m = (a + b) / 2
            stack.append((m, b)); stack.append((a, m)); continue
        out.append(b)
    return np.array(out[:-1] if closed else out)


def holonomy(P, gens, k, closed=True):
    """(смен ранжировки, матрица в конвенции кода, матрица направленного подъёма)."""
    P = refine(P, closed)
    perms = [tuple(np.argsort(r)) for r in P] + ([tuple(np.argsort(P[0]))] if closed else [])
    d = gens[0].shape[0]
    Mc = np.eye(d, dtype=complex); Mi = np.eye(d, dtype=complex); fl = 0
    for t in range(1, len(perms)):
        p0, p1 = perms[t - 1], perms[t]
        if p0 == p1:
            continue
        i, j = [q for q in range(k) if p0[q] != p1[q]]
        Mc = Mc @ transposition(gens, i, j)                       # по позициям, как в коде
        Mi = Mi @ (gens[p0[i]] - gens[p0[j]]) / np.sqrt(2)        # по меткам, lower → higher
        fl += 1
    return fl, Mc, Mi


def sign(M):
    I = np.eye(len(M))
    if np.allclose(M, I): return +1
    if np.allclose(M, -I): return -1
    return 0


def pierced(c, a, b, r, eqs):
    """Сколько страт (две линейные формы = 0) протыкают диск {c + s·a + t·b, s²+t² < r²}."""
    n = 0
    for (i, j), (l, m) in eqs:
        A = np.array([[a[i] - a[j], b[i] - b[j]], [a[l] - a[m], b[l] - b[m]]])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        s, t = np.linalg.solve(A, -np.array([c[i] - c[j], c[l] - c[m]]))
        n += (s * s + t * t < r * r)
    return n


def strata(k):
    pair2 = [((i, j), (l, m))
             for i, j in itertools.combinations(range(k), 2)
             for l, m in itertools.combinations(range(k), 2)
             if (i, j) < (l, m) and len({i, j, l, m}) == 4]        # {x_i=x_j, x_l=x_m}
    triple = [((i, j), (j, l)) for i, j, l in itertools.combinations(range(k), 3)]  # {x_i=x_j=x_l}
    return pair2, triple


def circle(c, a, b, r, steps=3000):
    th = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    return c[None, :] + r * (np.cos(th)[:, None] * a[None, :] + np.sin(th)[:, None] * b[None, :])


def frame(rs, k):
    a = rs.normal(size=k); a /= np.linalg.norm(a)
    b = rs.normal(size=k); b -= (b @ a) * a; b /= np.linalg.norm(b)
    return a, b


def random_loops(k, trials, rs):
    gens = clifford(k)
    pair2, triple = strata(k)
    agree = total = code_rule = 0; by = {}
    for _ in range(trials):
        r = rs.uniform(0.05, 0.3)
        c = rs.uniform(r + 0.02, 1 - r - 0.02, size=k)
        a, b = frame(rs, k)
        P = circle(c, a, b, r)
        fl, Mc, Mi = holonomy(P, gens, k)
        si, sc = sign(Mi), sign(Mc)
        if si == 0 or sc == 0:
            continue
        n2, n3 = pierced(c, a, b, r, pair2), pierced(c, a, b, r, triple)
        total += 1; agree += ((-1) ** n2 == si); code_rule += (sc == (-1) ** (fl // 2) * si)
        row = by.setdefault(min(n2, 3), [0, 0]); row[0] += 1; row[1] += (si == -1)
    print(f"\nk = {k}: {total} случайных петель, {len(pair2)} страт поворотов на π, {len(triple)} тройных")
    print(f"   предсказание (−1)^(π-страт в диске) совпало с направленным подъёмом: {agree} из {total}")
    print(f"   правило «код = (−1)^(смен/2) × направленный» выполнено:               {code_rule} из {total}")
    print(f"   {'π-страт в диске':>16} {'петель':>7} {'знак −1':>8}")
    for n2, (n, neg) in sorted(by.items()):
        print(f"   {(str(n2) if n2 < 3 else '3 и больше'):>16} {n:>7} {neg:>8}")


def exchanges(trials, rs):
    print("\nОБМЕН ДВУХ КОПИЙ (k=4): петля pA → pB → pA из двух полукругов, как в bridge.py")
    k = 4; gens = clifford(k); pair2, _ = strata(k)
    rows = {}; agree = total = 0
    for _ in range(trials):
        pA = rs.uniform(0.1, 0.9, size=k)
        dist = rs.choice([0.02, 0.05, 0.1, 0.2, 0.4, 0.6])
        d = rs.normal(size=k); d /= np.linalg.norm(d)
        pB = pA + dist * d
        if pB.min() < 0.02 or pB.max() > 0.98:
            continue
        c, r = (pA + pB) / 2, dist / 2
        e = rs.normal(size=k); e -= (e @ d) * d; e /= np.linalg.norm(e)
        P = circle(c, -d, e, r)
        if P.min() < 0 or P.max() > 1:
            continue
        fl, Mc, Mi = holonomy(P, gens, k)
        si = sign(Mi)
        if si == 0:
            continue
        total += 1; agree += ((-1) ** pierced(c, -d, e, r, pair2) == si)
        row = rows.setdefault(dist, [0, 0]); row[0] += 1; row[1] += (si == -1)
    print(f"   предсказание по зацеплению совпало: {agree} из {total}")
    print(f"   {'|pA−pB|':>8} {'обменов':>8} {'знак −1':>8}   доля")
    for dist, (n, neg) in sorted(rows.items()):
        print(f"   {dist:>8.2f} {n:>8} {neg:>8}   {neg / n:.2f}")
    print("   близкие копии — всегда +1; знак обмена = сколько осей поворота на π внутри петли")


if __name__ == "__main__":
    rs = np.random.default_rng(2026)
    random_loops(4, 400, rs)
    random_loops(5, 200, rs)
    exchanges(600, rs)
