"""Проверка 2026-09-09, результаты в docs/spin-doc.md §15.

Мост «спин ↔ статистика»: обмен двух копий даёт −1?

Берём пути обмена из bridge.exchange_path (как в работе) и поднимаем спинор
вдоль пути КАЖДОЙ копии двумя конвенциями (см. spin_check.py):
  CODE  — множитель по позициям, одинаковый в обе стороны (t² = −1)
  INVAR — направленный по меткам (обратный переход = обратный элемент)
Итог обмена — произведение переносов двух копий, применённое к опорному вектору.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from simulate import clifford, transposition  # noqa: E402
from bridge import exchange_path              # noqa: E402

K = 4
g = clifford(K)
I = np.eye(g[0].shape[0])


def vec(a, b):
    return (g[a] - g[b]) / np.sqrt(2)


def refine_open(P):
    out = [P[0]]
    stack = [(P[t + 1], P[t]) for t in range(len(P) - 2, -1, -1)]
    while stack:
        b, a = stack.pop()
        if (np.argsort(a) != np.argsort(b)).sum() > 2:
            m = (a + b) / 2
            stack.append((b, m)); stack.append((m, a)); continue
        out.append(b)
    return np.array(out)


def lift_open(P):
    """Перенос вдоль НЕзамкнутого пути. Возвращает (смен, M_code, M_invar)."""
    P = refine_open(P)
    perms = [tuple(np.argsort(r)) for r in P]
    Mc = np.eye(len(I), dtype=complex); Mi = np.eye(len(I), dtype=complex); fl = 0
    for t in range(1, len(perms)):
        p0, p1 = perms[t - 1], perms[t]
        if p0 == p1:
            continue
        i, j = [q for q in range(K) if p0[q] != p1[q]]
        Mc = Mc @ transposition(g, i, j)
        Mi = Mi @ vec(p0[i], p0[j])
        fl += 1
    return fl, Mc, Mi


def sign_of(M):
    if np.allclose(M, I): return "+1"
    if np.allclose(M, -I): return "−1"
    return "не ±1"


rs = np.random.default_rng(0)
rows = {}
for trial in range(400):
    pA = rs.random(K)
    order = np.argsort(pA)
    kind = rs.integers(0, 2)
    if kind == 0:
        # та же ранжировка: слегка сдвинуть, не меняя порядка
        pB = pA + rs.normal(scale=0.01, size=K)
        if tuple(np.argsort(pB)) != tuple(order): continue
    else:
        # соседние по рангу шкалы меняются местами: минимальный обмен
        pos = rs.integers(0, K - 1)
        a, b = order[pos], order[pos + 1]
        pB = pA.copy(); pB[a], pB[b] = pA[b] + 0.005, pA[a] - 0.005
        if (pB.min() < 0) or (pB.max() > 1): continue
    A, B = exchange_path(pA, pB, steps=3000, k=K)
    if np.abs(np.diff(A, axis=0)).max() > 0.5 or np.abs(np.diff(B, axis=0)).max() > 0.5:
        continue  # путь обернулся через край куба (mod 1) — пропускаем
    fa, Mca, Mia = lift_open(A)
    fb, Mcb, Mib = lift_open(B)
    key = (fa + fb, sign_of(Mca @ Mcb), sign_of(Mia @ Mib))
    rows[key] = rows.get(key, 0) + 1

print(f"{'смен ранжировки (A+B)':>24} {'CODE':>6} {'INVAR':>6} {'случаев':>8}")
for (n, sc, si), c in sorted(rows.items()):
    print(f"{n:>24} {sc:>6} {si:>6} {c:>8}")
