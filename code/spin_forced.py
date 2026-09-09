"""Где в накрытии группы перестановок шкал сидит вынужденный −1. См. docs/spin-doc.md §16.

Перестановка g шкал действует на пространстве (сумма шкал = const) как элемент O(k−1):
чётная — поворот, нечётная — с отражением. Подъём в Pin(k): произведение u_ij=(e_i−e_j)/√2
по любому разложению g в транспозиции; подъём определён с точностью до знака ±L.
Если порядок m элемента g чётный, L^m не зависит от знака: этот знак ВЫНУЖДЕН.
Считаем L^m в двух сигнатурах (e² = −1 и e² = +1) для всех классов сопряжённости.

Запуск: python spin_forced.py            (k = 3, 4, 5, 6)
"""
import itertools
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from simulate import clifford  # noqa: E402   e_i^2 = -1


def cycle_type(g):
    seen, ct = set(), []
    for i in range(len(g)):
        if i in seen:
            continue
        j, n = i, 0
        while j not in seen:
            seen.add(j); j = g[j]; n += 1
        ct.append(n)
    return tuple(sorted(ct, reverse=True))


def transpositions(g):
    """g как список образов i -> g[i]. Разложение в транспозиции (i j) по меткам шкал."""
    g = list(g); out = []
    for i in range(len(g)):
        while g[i] != i:
            j = g[i]
            out.append((i, j))
            g[i], g[j] = g[j], g[i]
    return out


def lift(g, gens):
    L = np.eye(gens[0].shape[0], dtype=complex)
    for i, j in transpositions(g):
        L = L @ (gens[i] - gens[j]) / np.sqrt(2)
    return L


def sign_of_power(L, m):
    P = np.linalg.matrix_power(L, m)
    I = np.eye(len(P))
    if np.allclose(P, I): return "+1"
    if np.allclose(P, -I): return "−1"
    return "не ±1"


def table(k):
    gm = clifford(k)                 # e² = −1  (Pin−)
    gp = [1j * e for e in gm]        # e² = +1  (Pin+); антикоммутация сохраняется
    rows = {}
    for g in itertools.permutations(range(k)):
        ct = cycle_type(g)
        m = math.lcm(*ct)
        even = (sum(c - 1 for c in ct) % 2 == 0)
        Lm, Lp = lift(g, gm), lift(g, gp)
        r = rows.setdefault(ct, {"n": 0, "even": even, "m": m, "minus": set(), "plus": set()})
        r["n"] += 1
        r["minus"] |= {sign_of_power(Lm, m), sign_of_power(-Lm, m)}
        r["plus"] |= {sign_of_power(Lp, m), sign_of_power(-Lp, m)}
    print(f"\nk = {k}: пространство {k-1}-мерное, {math.factorial(k)} перестановок")
    print(f"   {'класс':>14} {'штук':>5} {'тип':>10} {'порядок':>8}   {'L^m (e²=−1)':>12}   {'L^m (e²=+1)':>12}   вывод")
    forced = 0
    for ct in sorted(rows, key=lambda c: (not rows[c]['even'], rows[c]['m'], c)):
        r = rows[ct]
        typ = "поворот" if r["even"] else "отражение"
        sm = "/".join(sorted(r["minus"])); sp = "/".join(sorted(r["plus"]))
        if r["m"] % 2 == 1:
            verdict = "знак выбирается (нечётный порядок)"
        elif sm == sp == "−1":
            verdict = "ВЫНУЖДЕННЫЙ −1" + ("" if r["even"] else " (но это отражение)")
            forced += r["even"]
        elif sm == sp:
            verdict = "вынужденный +1"
        else:
            verdict = "зависит от сигнатуры Pin±"
        print(f"   {str(ct):>14} {r['n']:>5} {typ:>10} {r['m']:>8}   {sm:>12}   {sp:>12}   {verdict}")
    print(f"   классов поворотов с вынужденным −1: {forced}")


if __name__ == "__main__":
    ks = [int(a) for a in sys.argv[1:]] or [3, 4, 5, 6]
    for k in ks:
        table(k)
    print("\nИтог: вынужденный −1 у поворотов появляется с k = 4 (повороты на π, класс (2,2));"
          "\nпри k = 3 вынужденных знаков нет, любой знак там — конвенция.")
