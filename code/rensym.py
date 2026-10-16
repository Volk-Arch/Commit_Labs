#!/usr/bin/env python3
"""ТЕСТ RENOU В ПОЛЯРНОЙ СИСТЕМЕ С ПОЛЕМ ЗАКРУЧЕННОГО СРЕЗА.

Сеть: источник 1 → Алиса и Боб, источник 2 (такой же, вторая копия) → Боб и Чарли; Боб — идеальное измерение
в базисе Белла (руками, как раньше), Алиса — 3 настройки, Чарли — 6. Комплексная КМ 6√2 = 8.485, вещественная ≤ 7.6605,
опыт 7.83 ± 0.03.
Всё остальное — из системы: нечётная накачка (заряд 2q) → распад (бозонная вершина, дочери с зарядом q) → захват
в A₁, A₂ / B₁ = P(A₁), B₂ = P(A₂); поле — фазы ног (B/2)(T_a x_b − x_a T_b) (fieldtwist.py) на всех ногах, включая
световоды. Приборы Алисы — делители кристалла Y (полярное правило, вещественная матрица V) с фазами поля на
световодах A₁→Y, A₂→Y; у Чарли — зеркальные приборы (там поле даёт сопряжённые фазы). Плюс прямое наблюдение пути (Z).
Пластинок нет: единственный источник комплексности — поле."""
import sys, math, itertools, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
from closedsym import ClosedSym, CH, mink
from renou import BELL, local_search, Z, A_opt, C_opt, phip
sys.argv = _a
COND = 0.2
NMAX = int(__import__("os").environ.get("NMAX", 1500))
S = ClosedSym(); X = S.X; n = S.n; w = S.w; R = S.R
K = 2
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)
quads = []
for A1 in [k for k in range(n) if 1.0 <= Xa[k, 0] <= 5.0 and 0.2 <= Xa[k, 1] <= 3.5]:
    A2 = CH.img(w, A1, H, Hs)
    if A2 < 0: continue
    quads.append((A1, A2, int(S.Pimg[A1]), int(S.Pimg[A2])))
rec = sorted({k for q in quads for k in q})
odd = {j: np.sign(X[j, 1]) for j in range(n) if abs(X[j, 0] + 11) <= 0.45 and abs(X[j, 1]) <= 3.0 and X[j, 1] != 0}
strip = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
outl = [[] for _ in range(n)]
for (a, b, kd) in S.edges:
    if kd == 0: outl[a].append(b)
def fph(a, b, B): return np.exp(1j * B / 2 * (X[a, 0] * X[b, 1] - X[a, 1] * X[b, 0]))

def devices(A1, A2):
    out = []
    for Y in np.nonzero(R[A1] & R[A2])[0]:
        if Xa[Y, 0] - max(Xa[A1, 0], Xa[A2, 0]) > 2.5: continue
        din = np.array([X[Y] - X[A1], X[Y] - X[A2]]); ds = outl[Y]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                dout = np.array([X[ds[i]] - X[Y], X[ds[j]] - X[Y]])
                Kk = np.exp(-mink(dout[:, None, :], din[None, :, :]) / S.m0); U_, s_, Vt = np.linalg.svd(Kk)
                if s_.min() < COND * s_.max(): continue
                out.append((int(Y), U_ @ Vt))
    return out

def obs_of(T): return T.conj().T @ Z @ T
SX = np.array([[0, 1], [1, 0]]); SY = np.array([[0, -1j], [1j, 0]])
def bloch(o): return np.array([np.trace(o @ SX).real, np.trace(o @ SY).real, np.trace(o @ Z).real]) / 2

def cells_from(a1, a2):
    """вещественные ячейки кристалла: событие Y в будущем a1, a2; выходы — пары звеньев Y→(u, v)"""
    out = []
    for Y in np.nonzero(R[a1] & R[a2])[0]:
        if Xa[Y, 0] - max(Xa[a1, 0], Xa[a2, 0]) > 2.0: continue
        din = np.array([X[Y] - X[a1], X[Y] - X[a2]]); ds = outl[Y]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                dout = np.array([X[ds[i]] - X[Y], X[ds[j]] - X[Y]])
                Kk = np.exp(-mink(dout[:, None, :], din[None, :, :]) / S.m0); U_, s_, Vt = np.linalg.svd(Kk)
                if s_.min() < COND * s_.max(): continue
                out.append((int(Y), ds[i], ds[j], U_ @ Vt))
    return out

def devices2(A1, A2, B, nmax=4000, seed=0):
    """приборы из одной и двух ячеек (интерферометр Маха–Цендера из событий кристалла), фазы поля на всех световодах.
    Возвращает наблюдаемые Алисы и зеркальные (Чарли) — одинаковые устройства, отражённые P."""
    rs = np.random.default_rng(seed); P = S.Pimg
    def Tdev(chain, sgn):
        a1, a2 = chain[0]; T = np.eye(2, dtype=complex); prev = (a1, a2)
        for (Y, u, v, V) in chain[1:]:
            if sgn < 0: Yp, up, vp = P[Y], P[u], P[v]; p1, p2 = P[prev[0]], P[prev[1]]
            else: Yp, up, vp, p1, p2 = Y, u, v, prev[0], prev[1]
            T = V @ np.diag([fph(p1, Yp, B), fph(p2, Yp, B)]) @ T; prev = (u, v)
        return T
    c1 = cells_from(A1, A2); As, Cs = [Z], [Z]; seen = set()
    def add(chain):
        oa = obs_of(Tdev(chain, +1)); key = tuple(np.round(bloch(oa), 2))
        if key in seen: return
        seen.add(key); As.append(oa); Cs.append(obs_of(Tdev(chain, -1)))
    for c in c1: add([(A1, A2), c])
    tries = 0
    while len(As) < nmax and tries < 6 * nmax and c1:
        tries += 1; c = c1[rs.integers(len(c1))]
        c2 = cells_from(c[1], c[2])
        if not c2: continue
        add([(A1, A2), c, c2[rs.integers(len(c2))]])
    return As, Cs

for B in [float(b) for b in (sys.argv[1:] or ['0', '0.1', '0.2', '0.3', '0.5'])]:
    S.set_field(2 * B)
    _, _, _, r = S.run({k: np.array([v / math.sqrt(len(odd))]) for k, v in odd.items()}, record=set(range(n)))
    p = np.array([r[k][0] for k in range(n)])
    S.set_field(B)
    beta, emit, col = S.backward(rec)
    # проверка: прямой проход = обратный (одна дочь из события полосы до захвата)
    s0 = strip[len(strip) // 2]; _, _, _, rr = S.run({s0: np.array([1.0 + 0j])}, record=set(rec))
    chk = max(abs(rr[k][0] - emit[s0, col[k]]) for k in rec)
    rows = []
    for q in quads:
        iA = [col[q[0]], col[q[1]]]; iB = [col[q[2]], col[q[3]]]; M = np.zeros((2, 2), complex)
        for s in strip:
            lo = S.legs[s][1]
            if len(lo) < 2: continue
            M += p[s] * (beta[lo][:, iA].T @ (np.ones((len(lo), len(lo))) - np.eye(len(lo))) @ beta[lo][:, iB])
        nm = np.linalg.norm(M)
        if nm < 1e-300: continue
        Mn = M / nm; rows.append((2 * abs(np.linalg.det(Mn)), q, Mn))
    rows.sort(key=lambda t: -t[0])
    # «комплексность» пары: доля мнимой части после лучшего общего фазового поворота
    best = []
    for C, q, Mn in rows:
        if len(best) >= 3: break
        As, Cs = devices2(q[0], q[1], B, nmax=NMAX)
        if len(As) < 3: continue
        Ty = max(abs(np.trace(o @ np.array([[0, -1j], [1j, 0]])).real) / 2 for o in As)
        T = max(local_search(Mn, Mn, [BELL[k] for k in pm], As, Cs, restarts=8)[0] for pm in itertools.permutations(BELL))
        Ts_obs = max(local_search(Mn, Mn, [BELL[k] for k in pm], A_opt, C_opt, restarts=4)[0] for pm in itertools.permutations(BELL))
        Ts_st = max(local_search(phip, phip, [BELL[k] for k in pm], As, Cs, restarts=8)[0] for pm in itertools.permutations(BELL))
        best.append((T, C, Ty, q, len(As), Ts_obs, Ts_st))
    best.sort(key=lambda t: -t[0])
    T, C, Ty, q, na, To, Tst = best[0]
    print(f'B = qF = {B}: прямой = обратный до {chk:.1e}; пар {len(rows)}, C макс {rows[0][0]:.4f}; '
          f'лучший T = {T:.3f} (A₁ ({Xa[q[0], 0]:.2f}, {Xa[q[0], 1]:+.2f}), C {C:.3f}, настроек {na}, |y| наблюдаемых до {Ty:.3f}); '
          f'по парам T: {[round(b[0], 2) for b in best]}; диагностика: наше состояние + идеальные настройки {max(b[5] for b in best):.3f}, идеальное состояние + наши настройки {max(b[6] for b in best):.3f}  [вещ. граница 7.66, компл. 8.485]')
    sys.stdout.flush()
