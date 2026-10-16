#!/usr/bin/env python3
"""БЕЛЛ НА ПРИБОРАХ В ПОЛЯРНОЙ СИСТЕМЕ (closedsym.py).

Цепочка: нечётная накачка → распад в событиях среды (бозонная вершина) → захват в симметричную моду в A₁, A₂ / B₁ = P(A₁),
B₂ = P(A₂) → оповещение → световоды → ДЕЛИТЕЛЬ = событие кристалла Y по тому же полярному правилу (ядро
exp(−d_out·d_in/m₀) на векторах двух входящих световодов A₁→Y, A₂→Y и двух выходящих — звеньев Y→d₁, Y→d₂) →
детекторы d₁, d₂. Перед делителем — пластинка (фаза e^{iα} на световоде из A₂; пластинка = отрезок поля закрученного
среза, fieldplate.py). У Боба — зеркальный прибор P(·): его матрица совпадает с Алисиной точно (ковариантность).

Факт по пути: вещественный делитель, симметричный при обмене световодов (зеркальная геометрия), делить не может —
полярное разложение [[α, β], [β, α]] даёт 1 или обмен. Поэтому Y берётся вне оси зеркала (несимметричная геометрия).
Мера качества делителя — sin²2θ = 4|V₁₁|²|V₁₂|²; для синглета S макс = 2√2·sin²2θ + 2·cos²2θ."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
from closedsym import ClosedSym, CH, mink
from beamsplit import smax
sys.argv = _a
COND = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
M0 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
S = ClosedSym(m0=M0); X = S.X; n = S.n; w = S.w; R = S.R
K = 2
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)
quads = []
for A1 in [k for k in range(n) if 1.0 <= Xa[k, 0] <= 5.0 and 0.2 <= Xa[k, 1] <= 3.5]:
    A2 = CH.img(w, A1, H, Hs)
    if A2 < 0: continue
    quads.append((A1, A2, int(S.Pimg[A1]), int(S.Pimg[A2])))
rec = sorted({k for q in quads for k in q})
beta, emit, col = S.backward(rec)

def pump_field(inj):
    _, _, _, r = S.run(inj, record=set(range(n))); return np.array([r[k][0] for k in range(n)]).real
odd = {j: np.sign(X[j, 1]) for j in range(n) if abs(X[j, 0] + 11) <= 0.45 and abs(X[j, 1]) <= 3.0 and X[j, 1] != 0}
p = pump_field({k: np.array([v / math.sqrt(len(odd))]) for k, v in odd.items()})
strip = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
def Mq(q, srcs=strip):
    iA = [col[q[0]], col[q[1]]]; iB = [col[q[2]], col[q[3]]]; M = np.zeros((2, 2))
    for s in srcs:
        lo = S.legs[s][1]
        if len(lo) < 2: continue
        M += p[s] * (beta[lo][:, iA].T @ (np.ones((len(lo), len(lo))) - np.eye(len(lo))) @ beta[lo][:, iB])
    return M

outl = [[] for _ in range(n)]
for (a, b, kd) in S.edges:
    if kd == 0: outl[a].append(b)

def splitter(A1, A2):
    """лучший делитель Y для пары световодов из A1, A2: Y в будущем обоих, выходы — два звена Y→d."""
    best = None
    for Y in np.nonzero(R[A1] & R[A2])[0]:
        if Xa[Y, 0] - max(Xa[A1, 0], Xa[A2, 0]) > 2.5: continue
        din = np.array([X[Y] - X[A1], X[Y] - X[A2]])
        ds = outl[Y]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                dout = np.array([X[ds[i]] - X[Y], X[ds[j]] - X[Y]])
                Kk = np.exp(-mink(dout[:, None, :], din[None, :, :]) / S.m0)
                U_, s_, Vt = np.linalg.svd(Kk); V = U_ @ Vt
                if s_.min() < COND * s_.max(): continue          # делитель должен быть устойчив: ядро не почти вырождено
                devA = [A1, A2, int(Y), ds[i], ds[j]]; devB = [int(S.Pimg[k]) for k in devA]
                if any(R[a, b] or R[b, a] for a in devA for b in devB): continue      # установки разнесены
                bal = 4 * V[0, 0] ** 2 * V[0, 1] ** 2
                if best is None or bal > best[0]: best = (bal, int(Y), ds[i], ds[j], V)
    return best

def chsh(Mn, VA, VB, angles):
    E = np.zeros((len(angles), len(angles)))
    for ia, al in enumerate(angles):
        TA = VA @ np.diag([1, np.exp(1j * al)])
        for ib, be in enumerate(angles):
            TB = VB @ np.diag([1, np.exp(1j * be)]); P = np.abs(TA @ Mn @ TB.T) ** 2
            E[ia, ib] = P[0, 0] + P[1, 1] - P[0, 1] - P[1, 0]
    bS = 0
    for a0 in range(len(angles)):
        for a1 in range(len(angles)):
            u, v = E[a0] + E[a1], E[a0] - E[a1]
            bS = max(bS, u.max() + v.max(), -(u.min() + v.min()))
    return bS

rows = []
for q in quads:
    M = Mq(q); nm = np.linalg.norm(M)
    if nm < 1e-300: continue
    Mn = M / nm; C = 2 * abs(np.linalg.det(Mn))
    sp = splitter(q[0], q[1])
    if sp is None: continue
    rows.append((sp[0], C, nm ** 2, q, sp, Mn))
rows.sort(key=lambda r: -r[0])
print(f'четвёрок {len(quads)}, с делителем {len(rows)}; нечётная накачка: C от {min(r[1] for r in rows):.6f} до {max(r[1] for r in rows):.6f}')
print(f'качество лучшего делителя sin²2θ: {[round(r[0], 3) for r in rows[:8]]}')
ang = np.linspace(0, 2 * math.pi, 72, endpoint=False)
for bal, C, eta, q, sp, Mn in rows[:5]:
    _, Y, d1, d2, VA = sp
    # Бобов прибор — зеркало: проверяем, что матрица та же
    PY, Pd1, Pd2 = S.Pimg[Y], S.Pimg[d1], S.Pimg[d2]
    din = np.array([X[PY] - X[q[2]], X[PY] - X[q[3]]]); dout = np.array([X[Pd1] - X[PY], X[Pd2] - X[PY]])
    Kk = np.exp(-mink(dout[:, None, :], din[None, :, :]) / S.m0); U_, _, Vt = np.linalg.svd(Kk); VB = U_ @ Vt
    devA = [q[0], q[1], Y, d1, d2]; devB = [q[2], q[3], PY, Pd1, Pd2]
    caus = sum(bool(R[a, b] or R[b, a]) for a in devA for b in devB)
    S_dev = chsh(Mn, VA, VB, ang); S_real = chsh(Mn, VA, VB, np.array([0.0, math.pi]))
    S_pred = 2 * math.sqrt(2) * bal + 2 * (1 - bal)
    print(f'  A₁ ({Xa[q[0], 0]:.2f}, {Xa[q[0], 1]:+.2f}), делитель Y ({Xa[Y, 0]:.2f}, {Xa[Y, 1]:+.2f}): sin²2θ = {bal:.3f}; '
          f'|V_Боб − V_Алиса| = {np.abs(VB - VA).max():.1e}; C = {C:.6f}; '
          f'S на приборах = {S_dev:.4f} (формула {S_pred:.4f}); только вещественные пластинки (0, π): S = {S_real:.3f}; '
          f'причинных связей Алиса–Боб {caus}; |M|² {eta:.1e}')
    sys.stdout.flush()

# ================= БЕЗ ПЛАСТИНОК: настройка = выбор делителя (всё вещественно) =================
print('\nБЕЗ ПЛАСТИНОК И ПОЛЯ: настройка Алисы — какой из делителей кристалла Y стоит на её световодах; у Боба — зеркальные')
def options(A1, A2):
    out = []
    for Y in np.nonzero(R[A1] & R[A2])[0]:
        if Xa[Y, 0] - max(Xa[A1, 0], Xa[A2, 0]) > 2.5: continue
        din = np.array([X[Y] - X[A1], X[Y] - X[A2]]); ds = outl[Y]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                dout = np.array([X[ds[i]] - X[Y], X[ds[j]] - X[Y]])
                Kk = np.exp(-mink(dout[:, None, :], din[None, :, :]) / S.m0); U_, s_, Vt = np.linalg.svd(Kk)
                if s_.min() < COND * s_.max(): continue
                out.append((int(Y), ds[i], ds[j], U_ @ Vt))
    return out
res = []
for bal, C, eta, q, sp, Mn in rows:
    opts = options(q[0], q[1])
    devs = [[q[0], q[1], Y, d1, d2] for (Y, d1, d2, V) in opts]
    # варианты Алисы, разнесённые со всеми зеркальными вариантами Боба
    good = [i for i, dv in enumerate(devs) if not any(R[a, int(S.Pimg[b])] or R[int(S.Pimg[b]), a] for a in dv for b in dv)]
    if len(good) < 2: continue
    E = np.zeros((len(good), len(good)))
    for ia, i in enumerate(good):
        for ib, j in enumerate(good):
            P = np.abs(opts[i][3] @ Mn @ opts[j][3].T) ** 2; E[ia, ib] = P[0, 0] + P[1, 1] - P[0, 1] - P[1, 0]
    best = (0, None)
    for a0 in range(len(good)):
        for a1 in range(len(good)):
            u, v = E[a0] + E[a1], E[a0] - E[a1]
            for sgn in (1, -1):
                b0, b1 = (np.argmax(sgn * u), np.argmax(sgn * v)); val = sgn * (u[b0] + v[b1])
                if val > best[0]: best = (val, (a0, a1, b0, b1))
    val, (a0, a1, b0, b1) = best
    dA = devs[good[a0]] + devs[good[a1]]; dB = [int(S.Pimg[k]) for k in devs[good[b0]] + devs[good[b1]]]
    caus = sum(bool(R[a, b] or R[b, a]) for a in dA for b in dB)
    res.append((val, q, len(good), caus))
res.sort(key=lambda r: -r[0])
print(f'установок с ≥ 2 разнесёнными делителями: {len(res)}; S (всё вещественно, без отбора после оповещения): '
      f'лучший {res[0][0]:.4f}, медиана {np.median([r[0] for r in res]):.4f}, > 2 у {sum(r[0] > 2.0005 for r in res)}')
for val, q, ng, caus in res[:4]:
    print(f'  A₁ ({Xa[q[0], 0]:.2f}, {Xa[q[0], 1]:+.2f}): вариантов делителя {ng}; S = {val:.4f}; причинных связей между всеми 4 приборами Алиса–Боб {caus}')
