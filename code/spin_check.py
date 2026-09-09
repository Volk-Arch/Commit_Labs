"""Проверка 2026-09-09, результаты в docs/spin-doc.md §15.

Проверка: откуда берётся −1 за оборот в spin_loop.

Две конвенции подъёма перехода через стенку x_a = x_b:
  CODE  — как в simulate.py: множитель (e_i − e_{i+1})/√2 по ПОЗИЦИЯМ в ранжировке,
          один и тот же в обе стороны перехода (t² = −1).
  INVAR — направленный: множитель (e_lower − e_higher)/√2 по МЕТКАМ шкал, где lower —
          шкала, которая была меньше ДО перехода. Обратный переход даёт обратный
          элемент, поэтому «туда-обратно» = +1, и значение — гомотопический инвариант
          петли в дополнении к стратам коразмерности 2.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from simulate import clifford, transposition  # noqa: E402


def vec(gens, a, b):
    return (gens[a] - gens[b]) / np.sqrt(2)


def refine(P):
    """Если за один шаг ранжировка меняется больше чем одной транспозицией,
    делим шаг пополам (путь между соседними точками — отрезок), пока каждая
    смена не станет одиночной. Так ни одно пересечение стенки не теряется."""
    out = [P[0]]
    stack = [(P[(t + 1) % len(P)], P[t]) for t in range(len(P) - 1, -1, -1)]
    guard = 0
    while stack:
        b, a = stack.pop()
        pa, pb = np.argsort(a), np.argsort(b)
        if (pa != pb).sum() > 2:
            m = (a + b) / 2
            stack.append((b, m)); stack.append((m, a))
            guard += 1
            if guard > 10**6:
                raise RuntimeError("не сходится")
            continue
        out.append(b)
    return np.array(out[:-1])


def evaluate(P, gens, k):
    """Идём по замкнутому пути P (строки — точки в [0,1]^k). Возвращает
    (число смен ранжировки, множитель CODE, множитель INVAR)."""
    P = refine(P)
    perms = [tuple(np.argsort(row)) for row in P] + [tuple(np.argsort(P[0]))]
    d = gens[0].shape[0]
    Mc = np.eye(d, dtype=complex)
    Mi = np.eye(d, dtype=complex)
    flips = 0
    for t in range(1, len(perms)):
        p0, p1 = perms[t - 1], perms[t]
        if p0 == p1:
            continue
        pos = [i for i in range(k) if p0[i] != p1[i]]
        if len(pos) != 2:
            raise RuntimeError(f"вырожденный шаг: {pos}")
        i, j = pos
        Mc = Mc @ transposition(gens, i, j)          # позиции, как в коде
        lower, higher = p0[i], p0[j]                 # метки: lower была меньше
        Mi = Mi @ vec(gens, lower, higher)           # направленный подъём
        flips += 1
    v = np.zeros(d, dtype=complex); v[0] = 1
    sc = complex(np.round((Mc @ v) @ v.conj(), 6))
    si = complex(np.round((Mi @ v) @ v.conj(), 6))
    return flips, sc, si


def fmt(z):
    if abs(z + 1) < 1e-6: return "−1"
    if abs(z - 1) < 1e-6: return "+1"
    return f"{z}"


def circle(center, a, b, radius, steps=4000, turns=1):
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    b = np.asarray(b, float); b = b - (b @ a) * a; b = b / np.linalg.norm(b)
    th = np.linspace(0, 2 * np.pi * turns, steps * turns, endpoint=False)
    P = np.asarray(center, float)[None, :] + radius * (np.cos(th)[:, None] * a[None, :]
                                                      + np.sin(th)[:, None] * b[None, :])
    assert P.min() >= 0 and P.max() <= 1, "путь вышел за куб"
    return P


# ---------------------------------------------------------------- 1. соотношения
print("1. СООТНОШЕНИЯ В РЕАЛИЗАЦИИ (k=4)")
g = clifford(4)
t = [transposition(g, i, i + 1) for i in range(3)]
I = np.eye(4)
chk = lambda A, B: np.allclose(A, B)
print(f"   t_i^2 = -1            : {all(chk(x @ x, -I) for x in t)}")
print(f"   (t1 t2)^3 = -1        : {chk(np.linalg.matrix_power(t[0] @ t[1], 3), -I)}")
print(f"   (t1 t3)^2 = -1        : {chk(np.linalg.matrix_power(t[0] @ t[2], 2), -I)}")
print(f"   t1 t2 t1 = t2 t1 t2   : {chk(t[0] @ t[1] @ t[0], t[1] @ t[0] @ t[1])}")
print(f"   t1 t2 t1 = -t2 t1 t2  : {chk(t[0] @ t[1] @ t[0], -t[1] @ t[0] @ t[1])}")

# ---------------------------------------------------------------- 2. как в коде, по k
print("\n2. ОБХОД ВОКРУГ ДИАГОНАЛИ (как spin_loop: центр 0.5, радиус 0.3, случайная плоскость)")
print(f"   {'k':>2} {'смен':>5} {'k(k-1)':>7}   CODE (3 плоскости)     INVAR (3 плоскости)")
for k in range(2, 9):
    g = clifford(k)
    rc, ri, fl = [], [], []
    for seed in range(3):
        r = np.random.default_rng(seed)
        a = r.normal(size=k); b = r.normal(size=k)
        P = circle(np.full(k, 0.5), a, b, 0.3, steps=6000)
        f, sc, si = evaluate(P, g, k)
        fl.append(f); rc.append(fmt(sc)); ri.append(fmt(si))
    print(f"   {k:>2} {fl[0]:>5} {k*(k-1):>7}   {' '.join(f'{x:>4}' for x in rc):<22} {' '.join(f'{x:>4}' for x in ri)}")

# ---------------------------------------------------------------- 3. петли без поворота, k=4
print("\n3. ПЕТЛИ, В КОТОРЫХ НЕТ НИКАКОГО ОБОРОТА (k=4)")
g = clifford(4)
cases = [
    ("туда-обратно через одну стенку x1=x2 (ничего не охватывает)",
     circle([0.5 + 0.02 / np.sqrt(2), 0.5 - 0.02 / np.sqrt(2), 0.2, 0.8],
            [1, -1, 0, 0], [0, 0, 1, 0], 0.05)),
    ("две стенки x1=x2 и x3=x4 по очереди, их пересечение НЕ внутри",
     circle([0.3 + 0.045 / np.sqrt(2), 0.3 - 0.045 / np.sqrt(2), 0.7 + 0.045 / np.sqrt(2), 0.7 - 0.045 / np.sqrt(2)],
            [1, -1, 0, 0], [0, 0, 1, -1], 0.05)),
    ("вокруг пересечения x1=x2 & x3=x4 (страта типа m=2)",
     circle([0.3, 0.3, 0.7, 0.7], [1, -1, 0, 0], [0, 0, 1, -1], 0.05)),
    ("вокруг тройной точки x1=x2=x3 (страта типа m=3)",
     circle([0.3, 0.3, 0.3, 0.8], [1, -1, 0, 0], [1, 1, -2, 0], 0.05)),
    ("вокруг диагонали, но ДВА оборота",
     circle(np.full(4, 0.5), np.random.default_rng(0).normal(size=4),
            np.random.default_rng(1).normal(size=4), 0.3, steps=6000, turns=2)),
]
for name, P in cases:
    f, sc, si = evaluate(P, g, 4)
    print(f"   {name}")
    print(f"      смен ранжировки {f:>3}   CODE {fmt(sc):>3}   INVAR {fmt(si):>3}")
