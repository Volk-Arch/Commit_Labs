"""Гравитация как деформация кристалла. Гладко согнуть плоский кристалл под Шварцшильда
нельзя (кривизна + замедление времени не интегрируются). Поэтому кристалл собирается из
лоскутов: в каждой ячейке (t, r, φ) свой кусок золотого кристалла, выложенный в МЕСТНОЙ
системе отсчёта (e_t = ∂_t/√A, e_r = √A·∂_r, e_φ), со своим случайным сдвигом и бустом.
Швы между лоскутами — дефекты (в теории упругости кривизна = плотность дисклинаций).
Местный «раньше» — конус Шварцшильда, как раньше. Плотность в собственном объёме та же.
Проверка: трубка самых длинных цепочек против точной орбиты (дуга 135°, R = 8M)."""
import sys, math, time, json, numpy as np
sys.path.insert(0, '.')
from chains import local_edges, longest_from, longest_to
from gold4 import coord_values, ST, SI
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 4; CELL = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
MODE = sys.argv[3] if len(sys.argv) > 3 else 'local'        # local — лоскуты по местной системе; global — один кристалл без деформации
WF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0     # ширина размытого края тени
R = 8.0; RHO = 20.0; EPS = 1.0; DPHI = 0.75 * math.pi; FR = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
TK = DPHI * math.sqrt(R ** 3); TJ = TK * FR; T = TJ[-1] + 1
rlo, rhi = 0.55 * R, 1.45 * R; a0, a1 = -math.radians(25), DPHI + math.radians(25)
r_win = (RHO * 5 * math.sqrt(5) * 3 / (4 * math.pi)) ** (1 / 3)
def Af(r): return 1 - 2 / r
def ok(mid, d):
    r = np.hypot(mid[:, 1], mid[:, 2]); A = Af(r)
    dr = (mid[:, 1] * d[:, 1] + mid[:, 2] * d[:, 2]) / r; rdphi = (mid[:, 1] * d[:, 2] - mid[:, 2] * d[:, 1]) / r
    return (A > 0) & (A * d[:, 0] ** 2 > dr ** 2 / A + rdphi ** 2)
from chainspeed3 import prob as fprob
def _rf(w):
    if w == 0: return r_win
    ss = np.linspace(0, 3 * r_win, 20001); lo_, hi_ = 0.5 * r_win, 1.5 * r_win
    for _ in range(60):
        mid = 0.5 * (lo_ + hi_); d = np.trapezoid(fprob(ss, mid, w) * 3 * ss ** 2, ss) / r_win ** 3
        (lo_, hi_) = (mid, hi_) if d < 1 else (lo_, mid)
    return mid
RF = _rf(WF); RMAX = RF * (1 + WF)
def crystal_block(rng, half, center=None):
    """кусок кристалла (с размытым краем WF) в кубе [−half, half]³ своей системы после буста 0.8 и поворота"""
    eta = 0.8; th = rng.uniform(0, 2 * math.pi); n = np.array([math.cos(th), math.sin(th)])
    B = np.eye(3); ch, sh_ = math.cosh(eta), math.sinh(eta)
    B[0, 0] = ch; B[0, 1:] = sh_ * n; B[1:, 0] = sh_ * n; B[1:, 1:] = np.eye(2) + (ch - 1) * np.outer(n, n)
    hv = np.full(3, half) if np.ndim(half) == 0 else np.asarray(half)
    L = hv.max() * (ch + sh_) + 1; sh = rng.uniform(0, 300, 3)
    t, tp = coord_values((sh[0] - L) / ST, (sh[0] + L) / ST, RMAX / SI); vals = [(t * ST - sh[0], (tp * SI) ** 2)]
    for k in (1, 2):
        cc, cp = coord_values(sh[k] - L, sh[k] + L, RMAX); vals.append((cc - sh[k], cp ** 2))
    P = np.zeros((1, 3)); W = np.zeros(1)
    for k, (cc, w2) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(cc), 0); W = np.repeat(W, len(cc))
        P[:, k] = np.tile(cc, m); W += np.tile(w2, m); keep = W <= RMAX ** 2; P, W = P[keep], W[keep]
        Pb = P @ B.T if k == 2 else None
    if WF > 0:
        keep = rng.random(len(P)) < fprob(np.sqrt(W), RF, WF); P = P[keep]
    P = P @ B.T
    return P[np.all(np.abs(P) <= hv, axis=1)]

def boost_mat(rng):
    eta = 0.8; th = rng.uniform(0, 2 * math.pi); n = np.array([math.cos(th), math.sin(th)])
    B = np.eye(3); ch, sh_ = math.cosh(eta), math.sinh(eta)
    B[0, 0] = ch; B[0, 1:] = sh_ * n; B[1:, 0] = sh_ * n; B[1:, 1:] = np.eye(2) + (ch - 1) * np.outer(n, n)
    return B
def global_block(c, half, B, sh, rng):
    """точки ОДНОГО глобального кристалла (буст B, сдвиг sh) в физическом кубе c ± half"""
    Bi = np.linalg.inv(B); cc0 = Bi @ c; L = half * 2.3 + 1
    t, tp = coord_values((cc0[0] + sh[0] - L) / ST, (cc0[0] + sh[0] + L) / ST, RMAX / SI); vals = [(t * ST - sh[0], (tp * SI) ** 2)]
    for k in (1, 2):
        cc, cp = coord_values(cc0[k] + sh[k] - L, cc0[k] + sh[k] + L, RMAX); vals.append((cc - sh[k], cp ** 2))
    P = np.zeros((1, 3)); W = np.zeros(1)
    for k, (cc, w2) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(cc), 0); W = np.repeat(W, len(cc))
        P[:, k] = np.tile(cc, m); W += np.tile(w2, m); keep = W <= RMAX ** 2; P, W = P[keep], W[keep]
    if WF > 0:
        # монетка края — детерминированная по положению (одна и та же для соседних кубов)
        h = np.sin(P @ np.array([12.9898, 78.233, 37.719]) + 0.0) * 43758.5453; u = h - np.floor(h)
        P = P[u < fprob(np.sqrt(W), RF, WF)]
    P = P @ B.T
    return P[np.all(np.abs(P - c) <= half, axis=1)]

def build(rng):
    if MODE == 'global':
        B = boost_mat(rng); sh = rng.uniform(0, 300, 3); out = []
        for t0 in np.arange(0, T, 4.0):
            for x0 in np.arange(-rhi, rhi, 4.0):
                for y0 in np.arange(-3.0, rhi, 4.0):
                    c = np.array([t0 + 2, x0 + 2, y0 + 2]); Q = global_block(c, 2.0, B, sh, rng)
                    if len(Q) == 0: continue
                    r = np.hypot(Q[:, 1], Q[:, 2]); ph = np.arctan2(Q[:, 2], Q[:, 1])
                    out.append(Q[(Q[:, 0] >= 0) & (Q[:, 0] <= T) & (r > rlo) & (r < rhi) & (ph > a0) & (ph < a1)])
        Q = np.vstack(out); _, u = np.unique(Q.round(9), axis=0, return_index=True); return Q[np.sort(u)]
    out = []
    # ячейки: полоса по r шириной s(r), в ней по t шаг s, по φ дуга ~s.
    # MODE local — лоскуты по местной системе; flat — та же нарезка, но без поворота (A = 1);
    # adapt — местная система и размер s(r) = CELL·(r/8)^1.5 (мельче у дыры, где кривизна больше)
    size = (lambda r: CELL * (r / 8.0) ** 1.5) if MODE == 'adapt' else (lambda r: CELL)
    r0 = rlo
    while r0 < rhi:
        sz = size(r0 + 0.5 * size(r0)); rc = r0 + sz / 2
        nph = max(1, int(math.ceil((a1 - a0) * rc / sz))); dph = (a1 - a0) / nph
        A = Af(rc) if MODE in ('local', 'adapt') else 1.0
        for t0 in np.arange(0, T, sz):
            for k in range(nph):
                p0 = a0 + (k + 0.5) * dph; tc = t0 + sz / 2
                half = 0.5 * max(sz * math.sqrt(A), sz / math.sqrt(A), rc * dph) * 1.05
                Q = crystal_block(rng, half)
                er = np.array([math.cos(p0), math.sin(p0)]); ep = np.array([-math.sin(p0), math.cos(p0)])
                tt = tc + Q[:, 0] / math.sqrt(A); rr = Q[:, 1] * math.sqrt(A); ss = Q[:, 2]
                x = rc * er[0] + rr * er[0] + ss * ep[0]; y = rc * er[1] + rr * er[1] + ss * ep[1]
                r = np.hypot(x, y); ph = np.arctan2(y, x)
                inside = (tt >= t0) & (tt < t0 + sz) & (r >= r0) & (r < r0 + sz) & (ph >= a0 + k * dph) & (ph < a0 + (k + 1) * dph)
                out.append(np.column_stack([tt, x, y])[inside])
        r0 += sz
    return np.vstack(out)
t0c = time.time(); allm = []
for s in range(NS):
    rng = np.random.default_rng(700 + s); P = build(rng); nw = len(P)
    special = np.array([[0, R, 0]] + [[tj, R * math.cos(DPHI), R * math.sin(DPHI)] for tj in TJ])
    P = np.vstack([special, P]); o = np.argsort(P[:, 0], kind='stable'); inv = np.empty_like(o); inv[o] = np.arange(len(o)); P = P[o]
    ei, ej = local_edges(P, ok, EPS); dA, _ = longest_from(P, int(inv[0]), ei, ej)
    rP = np.hypot(P[:, 1], P[:, 2]); phP = np.arctan2(P[:, 2], P[:, 1]); mids = []
    for k in range(len(TJ)):
        b = int(inv[1 + k]); dB = longest_to(P, b, ei, ej)
        tube = (dA >= 0) & (dB >= 0) & (dA + dB == dA[b]); w = tube & (np.abs(phP - DPHI / 2) < 0.15)
        mids.append(float(rP[w].mean()) if w.any() else float('nan'))
    allm.append(mids)
    area = 0.5 * (a1 - a0) * (rhi ** 2 - rlo ** 2)
    print(f'{MODE} край {WF} ячейка {CELL}: мир {s}: событий {nw} (плотность {nw/(T*area):.1f}), радиус трубки {np.round(mids,2)}   [{time.time()-t0c:.0f} с]'); sys.stdout.flush()
A = np.array(allm)
print(f'ИТОГ {MODE} край {WF} ячейка {CELL}: {np.round(np.nanmean(A,0),2).tolist()} ± {np.round(np.nanstd(A,0)/math.sqrt(len(A)),2).tolist()}  (точно 6.76, 7.38, 8.00, 8.62, 9.23)')
json.dump(dict(mode=MODE, cell=CELL, w=WF, mid=np.nanmean(A, 0).tolist(), err=(np.nanstd(A, 0) / math.sqrt(len(A))).tolist(), all=A.tolist()), open(f'/home/claude/data/bend_{MODE}_{CELL:g}_w{WF:g}.json', 'w'))
