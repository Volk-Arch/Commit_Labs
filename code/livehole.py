#!/usr/bin/env python3
"""ЖИВОЙ МИР: чёрная дыра на винтовом срезе (2+1, M = 1), строится слоями по времени.

Каждый слой Δt — новые события золотого кристалла на изогнутом срезе (как в bentorbit.py,
но вокруг всей дыры), связи с прошлым, и потоковый счёт по всему, что уже построено:
  * часы — самая длинная цепочка от начала мира до каждого события;
  * свет — будущее трёх вспышек в момент t = 0 (r = 3, 6, 11);
  * орбита — самая длинная цепочка из точки (t = 0, r = 8, φ = 0);
  * место события в окне (угол и глубина), положение в тени (x', y') — «винт».
После каждого слоя пишется кадр data/live/<run>/frame_NNNN.json (сжатый), в конце —
orbit.json. Кадры потом заливаются в базу страницы-обсерватории.

    python3 livehole.py ИМЯ T_КОНЦА [ΔT]
"""
import sys, os, math, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import cKDTree
_argv = sys.argv; sys.argv = sys.argv[:1]
from bentorbit import Hole2, project, ok, Af        # bentorbit читает argv при импорте
sys.argv = _argv
from bentcut import zt_range, ST, SI

NAME = sys.argv[1] if len(sys.argv) > 1 else 'test'
TEND = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
OM, RHO, EPS = 0.12, 10.0, 1.0
RLO, RHI = 2.8, 14.0
RW = (RHO * 5 * math.sqrt(5) * 3 / (4 * math.pi)) ** (1 / 3)
OUT = f'/home/claude/data/live/{NAME}'; os.makedirs(OUT, exist_ok=True)
rs = np.random.default_rng(2026)
H = Hole2(OM)
ETA = 0.8; GM, GV = math.cosh(ETA), math.sinh(ETA); TH = rs.uniform(0, 2 * math.pi)
ROT = np.array([[math.cos(TH), -math.sin(TH)], [math.sin(TH), math.cos(TH)]])
SH = np.concatenate([rs.uniform(0, 300, 3), rs.uniform(-3, 3, 3)])
TK8 = math.pi * 8 ** 1.5                                   # полоборота на r = 8 (кеплеровское время)
FLASH = [(3.0, math.pi / 2), (6.0, math.pi), (11.0, 1.5 * math.pi)]
SRC = (8.0, 0.0)
TARGETS = [(f * TK8, 8.0, math.pi) for f in (0.85, 1.0, 1.15)]

def slab_events(t0, t1):
    """события среза с t ∈ [t0, t1): (t, x, y), угол и глубина в окне, тень (x', y')."""
    corners = np.array([[ta, X, Y] for ta in (t0 - 3, t1 + 3) for X in (-RHI - 3, RHI + 3) for Y in (-RHI - 3, RHI + 3)])
    uw = corners[:, 1:] @ ROT; Bi = np.array([[GM, -GV], [-GV, GM]])
    tx = np.column_stack([corners[:, 0], uw[:, 0]]) @ Bi.T
    lo = np.array([tx[:, 0].min(), tx[:, 1].min(), uw[:, 1].min()]) + SH[:3]
    hi = np.array([tx[:, 0].max(), tx[:, 1].max(), uw[:, 1].max()]) + SH[:3]
    Amin, Amax = float(H.A(np.array(RHI + 0.5))), float(H.A(np.array(RLO - 0.3)))
    Bmin, Bmax = float(H.B(np.array(RLO - 0.3))), float(H.B(np.array(RHI + 0.5)))
    t, tq = zt_range(lo[0] / ST, hi[0] / ST, (Bmin - RW + SH[3]) / SI, (Bmax + RW + SH[3]) / SI); T, Tq = t * ST - SH[0], tq * SI - SH[3]
    x, xq = zt_range(lo[1], hi[1], -Amax - RW + SH[4], Amax + RW + SH[4]); x, xq = x - SH[1], xq - SH[4]
    y, yq = zt_range(lo[2], hi[2], -Amax - RW + SH[5], Amax + RW + SH[5]); y, yq = y - SH[2], yq - SH[5]
    def dint(v, a, b): return np.maximum(0, np.maximum(a - v, v - b))
    dT = dint(Tq, Bmin, Bmax) ** 2; r2 = RW * RW * 1.05; out = []
    for k0 in range(0, len(T), 8):
        it = np.arange(k0, min(k0 + 8, len(T)))
        ii = np.repeat(it, len(x)); jx = np.tile(np.arange(len(x)), len(it))
        ta = GM * T[ii] + GV * x[jx]
        m = (ta > t0 - 3.2) & (ta < t1 + 3.2) & (dT[ii] <= r2); ii, jx, ta = ii[m], jx[m], ta[m]
        if len(ii) == 0: continue
        sp = OM * (ta[:, None] + np.linspace(-3.2, 3.2, 9)[None, :])
        c = np.cos(sp); cl = np.minimum(Amin * c, Amax * c).min(1) - 0.1; ch = np.maximum(Amin * c, Amax * c).max(1) + 0.1
        W = dT[ii] + dint(xq[jx], cl, ch) ** 2; m = W <= r2; ii, jx, ta, W = ii[m], jx[m], ta[m], W[m]
        if len(ii) == 0: continue
        s_ = np.sin(sp[m]); sl = np.minimum(Amin * s_, Amax * s_).min(1) - 0.1; sh_ = np.maximum(Amin * s_, Amax * s_).max(1) + 0.1
        n0 = len(ii); pa = np.repeat(np.arange(n0), len(y)); jy = np.tile(np.arange(len(y)), n0)
        W2 = W[pa] + dint(yq[jy], sl[pa], sh_[pa]) ** 2; m = W2 <= r2; pa, jy = pa[m], jy[m]
        iT, iX = ii[pa], jx[pa]
        u = GM * x[iX] + GV * T[iT]; XY = np.column_stack([u, y[jy]]) @ ROT.T
        P = np.column_stack([ta[pa], XY, xq[iX], yq[jy], Tq[iT]])
        rr = np.hypot(P[:, 1], P[:, 2]); P = P[(rr > 1.0) & (rr < RHI + 3)]
        if len(P) == 0: continue
        tt, XX, n2 = project(H, P)
        acc = n2 <= RW * RW
        Q = np.column_stack([tt, XX]); r = np.hypot(Q[:, 1], Q[:, 2])
        keep = acc & (Q[:, 0] >= t0) & (Q[:, 0] < t1) & (r > RLO) & (r < RHI)
        if not keep.any(): continue
        S = H.surf(tt[keep], XX[keep]); d = P[keep] - S             # смещение от среза (в тени)
        wa = np.arctan2(d[:, 4], d[:, 3]); wd = np.sqrt(np.maximum(n2[keep], 0)) / RW
        out.append(np.column_stack([Q[keep], wa, wd, P[keep][:, 3], P[keep][:, 4]]))
    E = np.vstack(out) if out else np.zeros((0, 7))
    return E[np.argsort(E[:, 0], kind='stable')]

def special(t, r, ph): return np.array([t, r * math.cos(ph), r * math.sin(ph), 0, 0, 0, 0.0])

def main():
    t0c = time.time()
    # потоковое состояние: хвост последних событий (t > t_cur − EPS)
    tail = np.zeros((0, 7)); tail_id = np.zeros(0, np.int64)
    tail_dC = np.zeros(0, np.int64); tail_dS = np.zeros(0, np.int64); tail_rc = np.zeros(0, np.int8)
    allpos = []; allpred = []; nid = 0
    target_ids = {}; frames = 0; total = 0; links_total = 0
    rb = np.linspace(RLO, RHI, 9)
    t = 0.0
    while t < TEND - 1e-9:
        t1 = min(t + DT, TEND)
        E = slab_events(t, t1)
        spec = []
        if t == 0.0:
            spec += [special(0.0, *SRC)] + [special(0.0, r, ph) for r, ph in FLASH]
        for k, (tt_, r_, ph_) in enumerate(TARGETS):
            if t <= tt_ < t1: spec.append(special(tt_, r_, ph_))
        kind = np.zeros(len(E), np.int8)                          # 0 обычное, 1 источник, 2–4 вспышки, 5–7 цели
        if spec:
            ks = []
            if t == 0.0: ks = [1, 2, 3, 4]
            ks += [5 + k for k, (tt_, _, _) in enumerate(TARGETS) if t <= tt_ < t1]
            E = np.vstack([np.array(spec), E]); kind = np.concatenate([np.array(ks, np.int8), kind])
            o = np.argsort(E[:, 0], kind='stable'); E, kind = E[o], kind[o]
        n = len(E); ids = np.arange(nid, nid + n); nid += n
        # связи: кандидаты из хвоста и слоя
        C = np.vstack([tail[:, :3], E[:, :3]]); nt = len(tail)
        tree = cKDTree(C); pr = tree.query_pairs(r=EPS * 1.5, output_type='ndarray')
        if len(pr):
            i, j = pr[:, 0], pr[:, 1]; sw = C[i, 0] > C[j, 0]; a = np.where(sw, j, i); b = np.where(sw, i, j)
            k = b >= nt; a, b = a[k], b[k]
            d = C[b] - C[a]; mid = 0.5 * (C[b] + C[a])
            k = (d[:, 0] > 0) & (d[:, 0] < EPS) & ok(mid, d); a, b = a[k], b[k]
        else:
            a = b = np.zeros(0, int)
        links_total += len(a)
        o = np.argsort(b, kind='stable'); a, b = a[o], b[o]
        starts = np.searchsorted(b, np.arange(nt, nt + n)); ends = np.searchsorted(b, np.arange(nt, nt + n), side='right')
        dC = np.concatenate([tail_dC, np.full(n, -1, np.int64)]); dS = np.concatenate([tail_dS, np.full(n, -1, np.int64)])
        rc = np.concatenate([tail_rc, np.zeros(n, np.int8)]); predS = np.full(n, -1, np.int64)
        gid = np.concatenate([tail_id, ids])
        for q in range(n):
            y = nt + q; p = a[starts[q]:ends[q]]
            if E[q, 0] < 1.0: dC[y] = 0
            if kind[q] == 1: dS[y] = 0
            if 2 <= kind[q] <= 4: rc[y] |= 1 << (kind[q] - 2)
            if len(p):
                cc = dC[p]; m_ = cc.max()
                if m_ >= 0 and dC[y] < 0: dC[y] = m_ + 1
                ss = dS[p]; ks = ss.argmax()
                if ss[ks] >= 0 and dS[y] < 0: dS[y] = ss[ks] + 1; predS[q] = gid[p[ks]]
                rc[y] |= np.bitwise_or.reduce(rc[p])
        for q in np.nonzero(kind >= 5)[0]: target_ids[int(kind[q]) - 5] = int(ids[q])
        allpos.append(E[:, :3].astype(np.float32)); allpred.append(predS)
        # кадр
        nE = dC[nt:]; nS = dS[nt:]; nR = rc[nt:]
        r = np.hypot(E[:, 1], E[:, 2]); total += n
        sel = np.arange(n) if n <= 2600 else np.sort(rs.choice(n, 2600, replace=False))
        tm = 0.5 * (t + t1)
        rate = np.where(nE[sel] >= 0, nE[sel] / max(tm, 0.5), 0)
        dens = np.histogram(r, rb)[0] / (DT * math.pi * np.diff(rb ** 2)) / RHO
        rbins = np.digitize(r, rb) - 1
        crate = [float(np.mean(nE[(rbins == k) & (nE >= 0)]) / max(tm, 0.5)) if np.any((rbins == k) & (nE >= 0)) else None for k in range(8)]
        fronts = []
        for bit in range(3):
            m_ = (nR & (1 << bit)) > 0
            fronts.append(float(m_.mean()))
        ssel = sel[:700]
        fr = dict(i=frames, t0=round(t, 3), t1=round(t1, 3), n=int(n), total=int(total), sec=round(time.time() - t0c, 1),
                  links=round(links_total / max(total, 1), 2),
                  x=np.round(E[sel, 1] * 100).astype(int).tolist(), y=np.round(E[sel, 2] * 100).astype(int).tolist(),
                  wa=np.round((E[sel, 3] + math.pi) / (2 * math.pi) * 63).astype(int).tolist(),
                  wd=np.round(np.clip(E[sel, 4], 0, 1) * 15).astype(int).tolist(),
                  clk=np.round(np.clip(rate, 0, 25.5) * 10).astype(int).tolist(),
                  lit=nR[sel].astype(int).tolist(), ds=np.clip(nS[sel], -1, 32000).astype(int).tolist(),
                  sx=np.round(E[ssel, 5] * 100).astype(int).tolist(), sy=np.round(E[ssel, 6] * 100).astype(int).tolist(),
                  sr=np.round(r[ssel] * 10).astype(int).tolist(),
                  dens=np.round(dens, 4).tolist(), crate=[None if v is None else round(v, 4) for v in crate],
                  lit_frac=[round(v, 4) for v in fronts], shadow_dir=round(float((OM * tm) % (2 * math.pi)), 4))
        json.dump(fr, open(f'{OUT}/frame_{frames:04d}.json', 'w'), separators=(',', ':'))
        frames += 1
        # новый хвост
        Cn = np.vstack([tail, E]); keep = Cn[:, 0] > t1 - EPS
        tail = Cn[keep]; tail_id = gid[keep]; tail_dC = dC[keep]; tail_dS = dS[keep]; tail_rc = rc[keep]
        print(f'слой {frames - 1}: t ∈ [{t:.1f}, {t1:.1f}), событий {n}, всего {total}, связей/событие {links_total / max(total, 1):.2f}, '
              f'густота по r {np.round(dens, 3).tolist()}  [{time.time() - t0c:.0f} с]'); sys.stdout.flush()
        t = t1
    # орбиты: цепочки из источника к целям по указателям
    pos = np.vstack(allpos); pred = np.concatenate(allpred)
    orb = []
    for k, tid in sorted(target_ids.items()):
        path = [tid]
        while pred[path[-1]] >= 0 and len(path) < 100000: path.append(int(pred[path[-1]]))
        P = pos[np.array(path[::-1])]
        orb.append(dict(f=[0.85, 1.0, 1.15][k], t=np.round(P[:, 0], 2).tolist(), x=np.round(P[:, 1] * 100).astype(int).tolist(), y=np.round(P[:, 2] * 100).astype(int).tolist(),
                        reached=bool(len(path) > 1), length=len(path) - 1))
    json.dump(dict(orbits=orb, frames=frames, total=int(total), sec=round(time.time() - t0c, 1)), open(f'{OUT}/orbit.json', 'w'), separators=(',', ':'))
    print(f'готово: кадров {frames}, событий {total}; орбиты: {[(o["f"], o["length"], o["reached"]) for o in orb]}  [{time.time() - t0c:.0f} с]')

if __name__ == '__main__':
    main()
