#!/usr/bin/env python3
"""ПРАВИЛО ВЗАИМОДЕЙСТВИЯ: «излом» + «поворот с фазой» в полярном событии — инерция, масса-часы, норма.

Событие: ядро K_ji = exp(−κ_ji/m₀) · exp(i μ · sign(η_out − η_in)),
  κ = τ(d_in + d_out) − τ_in − τ_out ≥ 0 — излом пути в событии (0 — прямо; Лоренц-инвариант, у ног разной длины ничьих нет),
  μ — фаза за поворот (вправо +μ, влево −μ; аналог «поворот = i» шахматной доски Фейнмана, μ = π/2 — ровно i).
Событие = ближайшая изометрия (полярное разложение), лишнее — в тень. Зеркало P: sign → −sign ⇒ симметрия P∘сопряжение.
Меряем на посыпке Пуассона (без ничьих) и золоте:
  (1) инерция: квант по одной ноге — сохраняется ли скорость; доля в мире;
  (2) ЧАСЫ: квант из точки; для ног внутри конуса — фаза амплитуды против собственного времени τ от источника:
      ищем частоту m, при которой Σ w e^{i(arg a − m τ)} максимальна; сравниваем m по полосам быстроты точки (x/t):
      если m одна и та же — часы идут по собственному времени (лоренц-инвариантная масса); контроль — то же по t."""
import sys, math, os, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from momentum import poisson_world, mink
sys.argv = _a

class Rule:
    def __init__(self, X, m0, mu, tol=1e-14):
        n = len(X); self.X = X; self.n = n
        d = X[None, :, :] - X[:, None, :]; R = d[..., 0] > np.abs(d[..., 1]) + 1e-12
        Rf = R.astype(np.float32); L = R & ~((Rf @ Rf) > 0)
        I, J = np.nonzero(L); self.I, self.J = I, J; self.E = len(I); self.dv = X[J] - X[I]
        self.rap = np.arctanh(self.dv[:, 1] / self.dv[:, 0])
        ein = [[] for _ in range(n)]; eout = [[] for _ in range(n)]
        for e in range(self.E): eout[I[e]].append(e); ein[J[e]].append(e)
        self.eout = eout; self.order = np.argsort(X[:, 0], kind='stable'); self.legs = []
        tau = lambda v: np.sqrt(np.maximum(mink(v, v), 0))
        for k in range(n):
            li, lo = np.array(ein[k], int), np.array(eout[k], int)
            if len(li) and len(lo):
                so = self.dv[lo][:, None, :] + self.dv[li][None, :, :]
                kap = tau(so) - tau(self.dv[lo])[:, None] - tau(self.dv[li])[None, :]
                K = np.exp(-kap / m0) * np.exp(1j * mu * np.sign(self.rap[lo][:, None] - self.rap[li][None, :]))
                U, s, Vt = np.linalg.svd(K, full_matrices=False); keep = s > tol * s.max(); V = U[:, keep] @ Vt[keep]
            else: V = np.zeros((len(lo), len(li)), complex)
            self.legs.append((li, lo, V))

    def run(self, src, e0=None):
        amp = np.zeros(self.E, complex); res = 0.0
        for k in self.order:
            li, lo, V = self.legs[k]
            vin = amp[li]; vout = V @ vin if len(li) and len(lo) else np.zeros(len(lo), complex)
            res += (np.abs(vin) ** 2).sum() - (np.abs(vout) ** 2).sum(); amp[lo] = vout
            if k == src:
                if e0 is None: amp[lo] += 1 / math.sqrt(len(lo))
                else: amp[e0] += 1.0
        return amp, res

def inertia(W, srcs, dts=(2.0, 4.0, 8.0), eta=0.8):
    X = W.X; out = []
    for s in srcs:
        lo = np.array(W.eout[s])
        if not len(lo): continue
        e0 = lo[np.argmin(np.abs(W.rap[lo] - eta))]
        if abs(W.rap[e0] - eta) > 0.25: continue
        amp, _ = W.run(s, e0); t0, x0 = X[s]; row = []
        for dt in dts:
            t = t0 + dt; a, b = W.I, W.J; cr = (X[a, 0] < t) & (X[b, 0] >= t)
            xm = X[a, 1] + (X[b, 1] - X[a, 1]) * (t - X[a, 0]) / (X[b, 0] - X[a, 0]); p = np.abs(amp[cr]) ** 2; P = p.sum()
            row.append(((xm[cr] - x0) * p).sum() / P / dt if P > 0 else np.nan); row.append(P)
        out.append([math.tanh(W.rap[e0])] + row)
    return np.nanmean(np.array(out), 0) if out else None

def clock(W, src):
    amp, res = W.run(src); X = W.X; t0, x0 = X[src]
    b = W.J; dt = X[b, 0] - t0; dx = X[b, 1] - x0; tau = np.sqrt(np.maximum(dt ** 2 - dx ** 2, 0))
    ok = (tau > 1.0) & (np.abs(amp) > 0); w = np.abs(amp[ok]) ** 2; ph = np.angle(amp[ok]); tt = tau[ok]; rr = np.arctanh(np.clip(dx[ok] / dt[ok], -0.999, 0.999))
    ms = np.linspace(-3, 3, 1201)
    def best(sel, var):
        z = np.array([np.abs((w[sel] * np.exp(1j * (ph[sel] - m * var[sel]))).sum()) / w[sel].sum() for m in ms])
        return ms[np.argmax(z)], z.max()
    allm = best(np.ones(len(w), bool), tt); allt = best(np.ones(len(w), bool), dt[ok])
    bands = []
    for lo_, hi_ in ((-2.5, -1.0), (-1.0, -0.3), (-0.3, 0.3), (0.3, 1.0), (1.0, 2.5)):
        sel = (rr >= lo_) & (rr < hi_)
        if w[sel].sum() > 0.02 * w.sum(): bands.append((lo_, hi_, *best(sel, tt), *best(sel, dt[ok])))
    return res, allm, allt, bands

if __name__ == '__main__':
    m0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    rho, Lw = 4.0, 14.0
    worlds = [('Пуассон', poisson_world(rho, Lw, np.random.default_rng(0))), ('золото', CH.sym_world(rho, Lw).X)]
    for mu in (0.0, 0.3, math.pi / 2):
        for name, X in worlds:
            W = Rule(X, m0, mu)
            srcs = [k for k in range(W.n) if abs(X[k, 0] + 9) < 0.6 and abs(X[k, 1]) < 1.5][:30]
            iv = inertia(W, srcs)
            src = min(srcs, key=lambda k: abs(X[k, 1]))
            res, allm, allt, bands = clock(W, src)
            msg = f'μ = {mu:.2f}, {name:7s}: инерция (старт v {iv[0]:.2f}) → v {iv[1]:+.2f}/{iv[3]:+.2f}/{iv[5]:+.2f} при Δt 2/4/8, в мире {iv[2]:.2f}/{iv[4]:.2f}/{iv[6]:.2f}; ' if iv is not None else f'μ = {mu:.2f}, {name}: '
            msg += f'часы: по τ m = {allm[0]:+.2f} (согласие {allm[1]:.2f}), по t m = {allt[0]:+.2f} ({allt[1]:.2f}); по полосам быстроты (m_τ / m_t): ' + \
                   ' '.join(f'[{a:+.1f},{b:+.1f}] {m1:+.2f}/{m2:+.2f}' for a, b, m1, c1, m2, c2 in bands)
            print(msg); sys.stdout.flush()
