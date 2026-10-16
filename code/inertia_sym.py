#!/usr/bin/env python3
"""ИНЕРЦИЯ ИЗ СИММЕТРИИ: в золотом кристалле буст g — точная симметрия, правило «излом» (Лоренц-инвариант) с ней
согласовано ⇒ квант, пущенный по ноге быстроты kη₀, есть g^k-образ кванта, пущенного «стоя» (быстрота 0). Стоящий на
оси квант по зеркалу P в среднем не сдвигается ⇒ образ летит ровно со скоростью tanh(kη₀) — импульс сохраняется
ТОЧНО (по Нётер: от симметрии), а не статистически. Нарушить это может только край конечного мира.
Меряем: (1) отклонение амплитуд образа от амплитуд оригинала на соответствующих ногах (ковариантность) в зависимости от
времени; (2) скорость центра в системе образа ⟨x'⟩/Δt' (должна быть 0) и в лаборатории; (3) то же для посыпки Пуассона,
где точного буста нет: там образа нет, меряем только скорость. Разные размеры мира — чтобы увидеть роль края."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from interact import Rule
from momentum import poisson_world
sys.argv = _a
m0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1

def slices(W, amp, s, dts, frame_eta=0.0):
    X = W.X; c, sh = math.cosh(frame_eta), math.sinh(frame_eta)
    Xf = np.stack([c * X[:, 0] - sh * X[:, 1], -sh * X[:, 0] + c * X[:, 1]], 1)
    t0, x0 = Xf[s]; out = []
    for dt in dts:
        t = t0 + dt; a, b = W.I, W.J; cr = (Xf[a, 0] < t) & (Xf[b, 0] >= t)
        xm = Xf[a, 1] + (Xf[b, 1] - Xf[a, 1]) * (t - Xf[a, 0]) / (Xf[b, 0] - Xf[a, 0]); p = np.abs(amp[cr]) ** 2; P = p.sum()
        out.append((((xm[cr] - x0) * p).sum() / P / dt if P > 0 else np.nan, P))
    return out

for Lw in (16.0, 24.0):
    w = CH.sym_world(4.0, Lw); X = w.X; W = Rule(X, m0, 0.0)
    eid = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(W.I, W.J))}
    axis = [k for k in range(W.n) if abs(X[k, 1]) < 1e-9 and -Lw * 0.3 < X[k, 0] < -Lw * 0.02]
    print(f'мир Lw = {Lw}: событий {W.n}; стартов на оси {len(axis)}')
    for K in (1, 2):
        G, Gs = np.linalg.matrix_power(CH.G, K), np.linalg.matrix_power(CH.Gs, K)
        img = np.array([CH.img(w, k, G, Gs) for k in range(W.n)])
        rows = []
        for s in axis:
            lo = np.array(W.eout[s]); vert = [e for e in lo if abs(W.rap[e]) < 1e-9]
            if not vert or img[s] < 0: continue
            e0 = vert[0]; ge0 = eid.get((int(img[W.I[e0]]), int(img[W.J[e0]])), -1)
            if ge0 < 0: continue
            a0, _ = W.run(s, e0); a1, _ = W.run(int(img[s]), ge0)
            mp = np.array([eid.get((int(img[a]), int(img[b])), -1) if img[a] >= 0 and img[b] >= 0 else -1 for a, b in zip(W.I, W.J)])
            errs = []
            for dt in (2.0, 4.0, 8.0):
                sel = (mp >= 0) & (X[W.J, 0] <= X[s, 0] + dt) & (X[W.J, 0] > X[s, 0])
                errs.append(np.abs(a1[mp[sel]] - a0[sel]).max() / np.abs(a0).max() if sel.any() else np.nan)
            vr = slices(W, a0, s, (2.0, 4.0, 8.0)); vb = slices(W, a1, int(img[s]), (2.0, 4.0, 8.0), K * CH.ETA0)
            vl = slices(W, a1, int(img[s]), (2.0, 4.0, 8.0))
            rows.append((errs, [v for v, p in vr], [v for v, p in vb], [v for v, p in vl], [p for v, p in vl]))
        if not rows: continue
        E = np.nanmean([r[0] for r in rows], 0); VR = np.nanmean([r[1] for r in rows], 0); VB = np.nanmean([r[2] for r in rows], 0)
        VL = np.nanmean([r[3] for r in rows], 0); PL = np.nanmean([r[4] for r in rows], 0)
        print(f'  буст g^{K} (скорость tanh({K}η₀) = {math.tanh(K * CH.ETA0):.3f}), стартов {len(rows)}: отклонение образа от оригинала '
              f'до Δt 2/4/8: {E[0]:.1e}/{E[1]:.1e}/{E[2]:.1e}; оригинал v = {VR[0]:+.3f}/{VR[1]:+.3f}/{VR[2]:+.3f}; '
              f'образ в своей системе v\' = {VB[0]:+.3f}/{VB[1]:+.3f}/{VB[2]:+.3f}; образ в лаборатории v = {VL[0]:.3f}/{VL[1]:.3f}/{VL[2]:.3f}, в мире {PL[2]:.2f}')
        sys.stdout.flush()
