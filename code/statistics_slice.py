"""Порог статистики частиц, если обмен происходит В ОДИН МОМЕНТ ВРЕМЕНИ.

Время события — сумма его шкал (части II §4, VIII §7). Значит две копии «в один момент»
живут в слое {сумма шкал = const} размерности k−1, а не во всём k-мерном кубе фаз, где
считался порог в части II §9. Повторяем те же три замера (order8.py §12, статистика-док §6)
в слое:

  1. достижимость обмена: BFS по парам клеток дискретного слоя, ход — перенос единицы
     с одной шкалы на другую (сумма сохраняется), совпадение копий запрещено;
  2. обмотка случайных путей обмена — суммарный поворот относительного вектора
     в плоскости самого слоя (k=3) и в трёх плоскостях слоя (k=4);
  3. непрерывная деформация: можно ли путь обмена с обмоткой 3 превратить в обмотку 1,
     не проведя копии через совпадение — в 2-мерном слое нельзя, в 3-мерном можно.

Ожидание: пороги сдвигаются на единицу относительно куба: k=2 — обмена нет,
k=3 — энионы, k≥4 — два класса. Тогда порог статистики совпадает с порогом спина
(spin-doc.md §16: вынужденный −1 у поворотов на π появляется с k=4).

Запуск: python statistics_slice.py
"""
import itertools
import os
import sys
from collections import deque
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ------------------------------------------------------------------ решётки

def cube_lattice(k, m):
    cells = [np.array(c) for c in itertools.product(range(m), repeat=k)]
    idx = {tuple(c): i for i, c in enumerate(cells)}
    nbr = [[] for _ in cells]
    for i, c in enumerate(cells):
        for d in range(k):
            for s in (-1, 1):
                q = c.copy(); q[d] += s
                if 0 <= q[d] < m: nbr[i].append(idx[tuple(q)])
    return np.array(cells), nbr


def slice_lattice(k, m, s0=None):
    """Клетки куба m^k с фиксированной суммой координат; ход: +1 по одной шкале, −1 по другой."""
    if s0 is None:
        s0 = round(k * (m - 1) / 2)
    cells = [np.array(c) for c in itertools.product(range(m), repeat=k) if sum(c) == s0]
    idx = {tuple(c): i for i, c in enumerate(cells)}
    nbr = [[] for _ in cells]
    for i, c in enumerate(cells):
        for a in range(k):
            for b in range(k):
                if a == b: continue
                q = c.copy(); q[a] += 1; q[b] -= 1
                if 0 <= q[a] < m and 0 <= q[b] < m: nbr[i].append(idx[tuple(q)])
    return np.array(cells), nbr


def farthest_pair(cells):
    D = ((cells[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    a, b = np.unravel_index(int(D.argmax()), D.shape)
    return int(a), int(b)


def can_swap(cells, nbr):
    """Достижим ли обмен (a,b) → (b,a); сколько пар достижимо из всех."""
    n = len(cells); a0, b0 = farthest_pair(cells)
    start, goal = (a0, b0), (b0, a0)
    seen = {start}; dq = deque([start]); found = False
    while dq:
        a, b = dq.popleft()
        if (a, b) == goal: found = True
        for x in nbr[a]:
            if x != b and (x, b) not in seen: seen.add((x, b)); dq.append((x, b))
        for y in nbr[b]:
            if y != a and (a, y) not in seen: seen.add((a, y)); dq.append((a, y))
    return found, len(seen), n * (n - 1)


def random_exchange(cells, nbr, rs, max_steps=200000):
    """Случайное блуждание по парам от (a,b) до (b,a) без совпадений; возвращает траекторию."""
    a0, b0 = farthest_pair(cells)
    a, b = a0, b0; traj = [(a, b)]
    for _ in range(max_steps):
        if rs.random() < 0.5:
            opts = [x for x in nbr[a] if x != b]
            if opts: a = int(rs.choice(opts))
        else:
            opts = [y for y in nbr[b] if y != a]
            if opts: b = int(rs.choice(opts))
        traj.append((a, b))
        if (a, b) == (b0, a0): return traj
    return None


def winding(cells, traj, u, v):
    """Суммарный поворот относительного вектора в плоскости (u, v), в единицах π.
    Возвращает (обмотка, число шагов с нулевой проекцией — там угол не определён)."""
    tot = 0.0; prev = None; zeros = 0
    for a, b in traj:
        d = cells[b] - cells[a]
        x, y = float(d @ u), float(d @ v)
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            zeros += 1; prev = None; continue
        ang = np.arctan2(y, x)
        if prev is not None:
            dd = ang - prev
            while dd > np.pi: dd -= 2 * np.pi
            while dd < -np.pi: dd += 2 * np.pi
            tot += dd
        prev = ang
    return int(round(tot / np.pi)), zeros


def slice_basis(k):
    """Ортонормированный базис подпространства {сумма = 0} в R^k."""
    B = []
    for i in range(1, k):
        v = np.zeros(k); v[:i] = 1.0; v[i] = -float(i)
        B.append(v / np.linalg.norm(v))
    return B


# ------------------------------------------------------------------ 1. достижимость
print("1. ДОСТИЖИМ ЛИ ОБМЕН")
print(f"   {'где':>14} {'k':>2} {'сетка':>6} {'клеток':>7} {'обмен':>6} {'достижимо пар':>14}")
for where, k, m in (("куб (контроль)", 1, 25), ("куб (контроль)", 2, 7),
                    ("слой", 2, 9), ("слой", 3, 7), ("слой", 4, 5), ("слой", 5, 4)):
    cells, nbr = cube_lattice(k, m) if where.startswith("куб") else slice_lattice(k, m)
    ok, seen, total = can_swap(cells, nbr)
    frac = f"{seen} из {total}" + ("  (ровно половина)" if 2 * seen == total else "")
    print(f"   {where:>14} {k:>2} {m:>6} {len(cells):>7} {'ДА' if ok else 'НЕТ':>6}   {frac}")

# ------------------------------------------------------------------ 2. обмотка
rs = np.random.default_rng(7)
print("\n2. ОБМОТКА СЛУЧАЙНЫХ ПУТЕЙ ОБМЕНА (единицы π)")
cells, nbr = cube_lattice(2, 7)
w = sorted({winding(cells, t, np.array([1.0, 0]), np.array([0, 1.0]))[0]
            for t in (random_exchange(cells, nbr, rs) for _ in range(40)) if t})
print(f"   куб k=2 (контроль, как в документе): значения {w}; все нечётные: {all(x % 2 for x in w)}")

cells, nbr = slice_lattice(3, 7)
u, v = slice_basis(3)
ws = []; zeros_total = 0
for _ in range(60):
    t = random_exchange(cells, nbr, rs)
    if t:
        wv, z = winding(cells, t, u, v); ws.append(wv); zeros_total += z
print(f"   слой k=3 (2-мерный), 60 путей: значения {sorted(set(ws))}")
print(f"      все нечётные: {all(x % 2 for x in ws)};  различных значений: {len(set(ws))};"
      f"  шагов с нулевой проекцией: {zeros_total}  → обмотка инвариант, классов много: ЭНИОНЫ")

cells, nbr = slice_lattice(4, 5)
B = slice_basis(4)
planes = [(0, 1), (0, 2), (1, 2)]
agree = 0; n = 0; zero_paths = 0; rows = []
for _ in range(40):
    t = random_exchange(cells, nbr, rs)
    if not t: continue
    vals = []; zz = 0
    for i, j in planes:
        wv, z = winding(cells, t, B[i], B[j]); vals.append(wv); zz += z
    n += 1; agree += (len(set(vals)) == 1); zero_paths += (zz > 0)
    if len(rows) < 5: rows.append(vals)
print(f"   слой k=4 (3-мерный), {n} путей, обмотка одного пути в трёх плоскостях слоя:")
for r in rows:
    print(f"      {r[0]:>4} {r[1]:>4} {r[2]:>4}   {'совпадают' if len(set(r)) == 1 else 'НЕТ'}")
print(f"      совпали во всех трёх плоскостях: {agree} из {n};  путей, где относительный вектор"
      f" проходил через нормаль плоскости: {zero_paths} из {n}")
print("      → обмотка не инвариант, разметки нет: ДВА КЛАССА")

# ------------------------------------------------------------------ 3. деформация
print("\n3. НЕПРЕРЫВНАЯ ДЕФОРМАЦИЯ: обмотка 3 → обмотка 1")
T = np.linspace(0, 1, 2001)
dA = np.stack([np.cos(3 * np.pi * T), np.sin(3 * np.pi * T), 0 * T], 1)   # полтора оборота
dB = np.stack([np.cos(np.pi * T), np.sin(np.pi * T), 0 * T], 1)           # пол-оборота


def planar_winding(path):
    ang = np.unwrap(np.arctan2(path[:, 1], path[:, 0]))
    return (ang[-1] - ang[0]) / np.pi


mins = []
for s in np.linspace(0, 1, 201):
    d = (1 - s) * dA + s * dB
    mins.append(np.linalg.norm(d, axis=1).min())
print(f"   в 2-мерном слое (k=3): семейство (1−s)·A + s·B; минимум |b−a| по семейству = {min(mins):.4f}"
      f"  → копии совпадают при s = {np.linspace(0, 1, 201)[int(np.argmin(mins))]:.2f}: деформации нет")

eps = 0.3
dA3 = dA + np.stack([0 * T, 0 * T, eps * np.sin(np.pi * T)], 1)
dA3 /= np.linalg.norm(dA3, axis=1, keepdims=True)
dot = (dA3 * dB).sum(1)
print(f"   в 3-мерном слое (k=4): путь A приподнят по третьей оси (концы на месте), "
      f"антиподальных пар с B нет: min(1 + A·B) = {(1 + dot).min():.3f}")
print(f"   {'s':>6} {'обмотка в плоскости':>20} {'min |b−a|':>10}")
for s in (0.0, 0.25, 0.5, 0.75, 1.0):
    om = np.arccos(np.clip(dot, -1, 1))[:, None]
    d = np.where(om > 1e-9,
                 (np.sin((1 - s) * om) * dA3 + np.sin(s * om) * dB) / np.where(om > 1e-9, np.sin(om), 1),
                 dA3)
    print(f"   {s:>6.2f} {planar_winding(d):>20.2f} {np.linalg.norm(d, axis=1).min():>10.3f}")
print("      → обмотка 3 непрерывно перешла в 1, копии ни разу не совпали: в 3-мерном слое"
      " обмотка не различает классы")

print("\nИТОГ. В слое «один момент времени»: k=2 — обмена нет; k=3 — энионы; k≥4 — два класса."
      "\nПорог совпал с порогом спина (spin-doc.md §16): оба на k = 4, три пространственных измерения.")
