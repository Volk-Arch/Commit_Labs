#!/usr/bin/env python3
"""МОСТ: спин из перестановок шкал (spin-doc) = спин 1/2 золотого кристалла 3+1.

1) Точные вращения золотого мира 3+1 (gold4.py): ортогональная матрица с элементами из Z[τ] должна иметь
   |a| ≤ 1 и |a'| ≤ 1 у каждого элемента ⇒ элементы только 0, ±1 ⇒ вращения мира — знаковые перестановки осей,
   собственных 24 = группа куба O. Проверяем на точках мира: все 48 знаковых перестановок оставляют мир на месте,
   а икосаэдрический и случайный поворот — нет.
2) O ≅ S₄: вращения куба переставляют 4 его большие диагонали — та же группа, что перестановки 4 шкал в spin-doc.
3) Подъём O в SU(2) — бинарная октаэдрическая группа 2O (48). Проверяем соотношения накрытия Шура
   t_i² = z, (t_i t_{i+1})³ = z, (t_i t_j)² = z (|i−j| > 1), z = −1: 2O — то же накрытие S̃₄, что в spin-doc.
   Но у куба транспозиции — ПОВОРОТЫ на 180° (det +1), а в spin-doc — отражения: там Pin, здесь Spin.
4) 4-мерное представление spin-doc (алгебра Клиффорда) раскладывается на два спиновых 2-мерных (по характерам).
5) Оборот на 2π (четыре поворота на 90°) даёт −1, два оборота +1."""
import sys, math, itertools, numpy as np
sys.path.insert(0, '/home/claude/bundle/bundle/code')
from gold4 import points_in_box, r_for_density, TAU
from cliff import clifford, transposition_op

# ---------- 1) точные вращения мира ----------
r = r_for_density(2.0); P = points_in_box([(-1.5, 1.5)] * 4, r)
from gold4 import coord_values
# тени точек: восстановим по координатам (c = a + bτ ⇒ c' однозначно по c на решётке): строим словарь из coord_values
def shadow_of(c, rint=r):
    a = np.round(c - np.floor(c)); return None
key = lambda p: tuple(np.round(p, 9))
S0 = {key(p) for p in P}
def inside(M):
    """доля точек мира (в центральном шаре), чей образ при пространственной матрице M тоже точка мира
    (сравниваем с миром в ящике побольше, чтобы край не мешал). Тень преобразуется сопряжённой матрицей M'."""
    Pbig = points_in_box([(-1.5, 1.5)] + [(-2.7, 2.7)] * 3, r); Sb = {key(p) for p in Pbig}
    m = np.linalg.norm(P[:, 1:], axis=1) < 1.4; Q = P[m].copy(); Q[:, 1:] = Q[:, 1:] @ M.T
    return np.mean([key(q) in Sb for q in Q]), m.sum()
signed = []
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1, -1), repeat=3):
        M = np.zeros((3, 3)); M[range(3), perm] = sg; signed.append(M)
ok = [inside(M)[0] for M in signed]
ico = 0.5 * np.array([[1, -TAU, 1 / TAU], [TAU, 1 / TAU, -1], [1 / TAU, 1, TAU]])
th = 0.7; rnd = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
print(f'1) мир 3+1: точек {len(P)}; знаковых перестановок 48: доля точек, перешедших в точки мира, мин {min(ok):.4f}; '
      f'икосаэдрический поворот (ортогонален: {np.allclose(ico @ ico.T, np.eye(3))}) {inside(ico)[0]:.4f}; поворот на 0.7 рад {inside(rnd)[0]:.4f}')

# ---------- 2) O ≅ S₄ через 4 большие диагонали ----------
O = [M for M in signed if np.linalg.det(M) > 0]
diag = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float)
def perm_of(M):
    out = []
    for d in diag:
        v = M @ d; out.append(int(np.argmin([min(np.linalg.norm(v - e), np.linalg.norm(v + e)) for e in diag])))
    return tuple(out)
perms = {perm_of(M) for M in O}
print(f'2) собственных вращений {len(O)}; различных перестановок диагоналей {len(perms)} (S₄ = 24) ⇒ O ≅ S₄: {len(perms) == 24}')

# ---------- 3) подъём в SU(2) ----------
def su2(M):
    """кватернион (как SU(2)) поворота M, знак — любой из двух"""
    w = math.sqrt(max(1 + np.trace(M), 0)) / 2
    if w > 1e-9:
        x = (M[2, 1] - M[1, 2]) / (4 * w); y = (M[0, 2] - M[2, 0]) / (4 * w); z = (M[1, 0] - M[0, 1]) / (4 * w)
    else:   # поворот на π: ось — собственный вектор с λ = 1
        vals, vecs = np.linalg.eig(M); ax = np.real(vecs[:, np.argmin(np.abs(vals - 1))]); ax /= np.linalg.norm(ax); x, y, z = ax
    sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1, -1])
    return w * np.eye(2) - 1j * (x * sx + y * sy + z * sz)
lifts = [su2(M) for M in O]
G = []
def add(U):
    for V in G:
        if np.abs(U - V).max() < 1e-9: return False
    G.append(U); return True
for U in lifts: add(U); add(-U)
frontier = list(G)
while frontier:
    new = []
    for A in frontier:
        for B in lifts:
            C = A @ B
            if add(C): new.append(C)
    frontier = new
print(f'3) группа подъёмов в SU(2): порядок {len(G)} (бинарная октаэдрическая 2O = 48)')
# транспозиции соседних диагоналей (0 1), (1 2), (2 3) — какие вращения их дают
def rot_for(p):
    for M in O:
        if perm_of(M) == p: return M
T = [rot_for(p) for p in [(1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)]]
print('   транспозиции диагоналей — повороты на угол', [round(math.degrees(math.acos((np.trace(M) - 1) / 2))) for M in T], 'град, det', [round(np.linalg.det(M)) for M in T])
best = None
for sg in itertools.product((1, -1), repeat=3):
    t = [s * su2(M) for s, M in zip(sg, T)]; I2 = np.eye(2); z = -I2
    rel = [np.abs(ti @ ti - z).max() for ti in t] + [np.abs(np.linalg.matrix_power(t[i] @ t[i + 1], 3) - z).max() for i in range(2)] + \
          [np.abs((t[0] @ t[2]) @ (t[0] @ t[2]) - z).max()]
    if best is None or max(rel) < best[0]: best = (max(rel), sg)
print(f'   соотношения накрытия Шура t² = z, (t_i t_{{i+1}})³ = z, (t_i t_j)² = z при z = −1: выполнены до {best[0]:.1e} (знаки {best[1]})')

# ---------- 4) представление spin-doc ----------
e = clifford(4); tc = [transposition_op(e, i, i + 1) for i in range(3)]
H = []
def add2(U):
    for V in H:
        if np.abs(U - V).max() < 1e-9: return False
    H.append(U); return True
for U in tc: add2(U)
fr = list(H)
while fr:
    nw = []
    for A in fr:
        for B in tc:
            C = A @ B
            if add2(C): nw.append(C)
    fr = nw
# сопоставим элементы по словам: строим гомоморфизм через слова из генераторов и сравним характеры
words = {(): (np.eye(4, dtype=complex), np.eye(2, dtype=complex))}
tq = [s * su2(M) for s, M in zip(best[1], T)]
frontier = [()]
while frontier and len(words) < 400:
    nw = []
    for w in frontier:
        A, B = words[w]
        for i in range(3):
            A2, B2 = A @ tc[i], B @ tq[i]
            if not any(np.abs(A2 - a).max() < 1e-9 for a, _ in words.values()): words[w + (i,)] = (A2, B2); nw.append(w + (i,))
    frontier = nw
consistent = all(abs(abs(np.trace(A)) - 0) < 9 for A, B in words.values())
ratios = [np.trace(A) / np.trace(B) for A, B in words.values() if abs(np.trace(B)) > 1e-9]
print(f'4) группа spin-doc (Клиффорд, 4 шкалы): порядок {len(H)}; сопоставлено слов {len(words)}; '
      f'след 4-мерного / след 2-мерного спинового: {sorted(set(np.round(np.real(ratios), 6)))} (2 ⇒ 4 = 2 ⊕ 2)')

# ---------- 5) оборот ----------
Rz = rot_for(perm_of(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])))
U = su2(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float))
print(f'5) поворот на 90° вокруг z, подъём в SU(2): 4 раза (оборот) = {np.round(np.linalg.matrix_power(U, 4)[0, 0].real, 6)}·1, '
      f'8 раз (два оборота) = {np.round(np.linalg.matrix_power(U, 8)[0, 0].real, 6)}·1')
par = {(len(w) % 2, round(float(np.real(np.trace(A) / np.trace(B))), 6)) for w, (A, B) in words.items() if abs(np.trace(B)) > 1e-9}
print(f'   (чётность слова, отношение следов): {sorted(par)} ⇒ 4 = (спин ½) ⊕ (спин ½ × знак перестановки) — две спиновые половины разной чётности, как у Дирака')
P2 = points_in_box([(-1.5, 1.5)] * 4, r_for_density(30.0)); print(f'   (проверка 1 на мире плотнее: точек {len(P2)})')
r = r_for_density(30.0); P = P2
ok = [inside(M)[0] for M in signed]
print(f'   знаковые перестановки: мин {min(ok):.4f}; икосаэдрический {inside(ico)[0]:.4f}; поворот 0.7 рад {inside(rnd)[0]:.4f}')
