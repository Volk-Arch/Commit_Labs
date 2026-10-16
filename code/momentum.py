#!/usr/bin/env python3
"""ИМПУЛЬС КВАНТА: летит ли квант, пущенный по одной ноге, дальше в ту же сторону (полярное событие, тень-сток).

Квант впрыскивается в ОДНУ выходную ногу события с быстротой η₀ (направленный старт). На срезах t = t₀ + Δ
меряем распределение |амплитуды|² по ногам, пересекающим срез: средняя скорость ⟨x − x₀⟩/Δ, доля, оставшаяся в мире,
и разброс. Сравниваем золотой кристалл с посыпкой Пуассона той же плотности (в `swerve-doc` у классической частицы
кристалл держал скорость, посыпка — нет). Скорость «из ноги» — tanh η₀."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a

import os
KERNEL = os.environ.get('KERNEL', 'минковский'); TOL = float(os.environ.get('TOL', '1e-9'))
def mink(a, b): return a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]

class Polar:
    def __init__(self, X, m0=2.0):
        n = len(X); self.X = X; self.n = n
        d = X[None, :, :] - X[:, None, :]; R = d[..., 0] > np.abs(d[..., 1]) + 1e-12
        Rf = R.astype(np.float32); L = R & ~((Rf @ Rf) > 0)
        I, J = np.nonzero(L); self.I, self.J = I, J; self.E = len(I); self.dv = X[J] - X[I]
        self.rap = np.arctanh(self.dv[:, 1] / self.dv[:, 0])
        ein = [[] for _ in range(n)]; eout = [[] for _ in range(n)]
        for e in range(self.E): eout[I[e]].append(e); ein[J[e]].append(e)
        self.order = np.argsort(X[:, 0], kind='stable'); self.legs = []
        for k in range(n):
            li, lo = np.array(ein[k], int), np.array(eout[k], int)
            if len(li) and len(lo):
                if KERNEL == 'единичный':      # только угол между ногами: exp(−(cosh Δη − 1)/m₀) — предпочитает «прямо»
                    uo = self.dv[lo] / np.sqrt(mink(self.dv[lo], self.dv[lo]))[:, None]; ui = self.dv[li] / np.sqrt(mink(self.dv[li], self.dv[li]))[:, None]
                    K = np.exp(-(mink(uo[:, None, :], ui[None, :, :]) - 1) / m0)
                elif KERNEL == 'излом':        # «излом» κ = τ(d_in + d_out) − τ_in − τ_out ≥ 0 (0 — продолжение прямо), есть длины ⇒ нет ничьих
                    so = self.dv[lo][:, None, :] + self.dv[li][None, :, :]
                    tau = lambda v: np.sqrt(np.maximum(mink(v, v), 0))
                    kap = tau(so) - tau(self.dv[lo])[:, None] - tau(self.dv[li])[None, :]
                    K = np.exp(-kap / m0)
                else:
                    K = np.exp(-mink(self.dv[lo][:, None, :], self.dv[li][None, :, :]) / m0)
                U, s, Vt = np.linalg.svd(K, full_matrices=False); keep = s > TOL * s.max(); V = U[:, keep] @ Vt[keep]
            else: V = np.zeros((len(lo), len(li)))
            self.legs.append((li, lo, V))
        self.eout = eout

    def run_leg(self, e0):
        amp = np.zeros(self.E); res = np.zeros(self.n); src = self.I[e0]
        for k in self.order:
            li, lo, V = self.legs[k]
            vin = amp[li]; vout = V @ vin if len(li) and len(lo) else np.zeros(len(lo))
            res[k] = (vin ** 2).sum() - (vout ** 2).sum()
            amp[lo] = vout
            if k == src: amp[e0] += 1.0
        return amp, res

def measure(W, e0, dts):
    amp, res = W.run_leg(e0); X = W.X; s = W.I[e0]; t0, x0 = X[s]
    out = []
    for dt in dts:
        t = t0 + dt; a, b = W.I, W.J; cr = (X[a, 0] < t) & (X[b, 0] >= t)
        if not cr.any(): out.append((np.nan,) * 3); continue
        xm = X[a, 1] + (X[b, 1] - X[a, 1]) * (t - X[a, 0]) / (X[b, 0] - X[a, 0])
        p = amp[cr] ** 2; P = p.sum()
        v = ((xm[cr] - x0) * p).sum() / P / dt if P > 0 else np.nan
        sd = math.sqrt(max(((xm[cr] - x0) ** 2 * p).sum() / P - (v * dt) ** 2, 0)) / dt if P > 0 else np.nan
        out.append((v, P, sd))
    return np.array(out)

def poisson_world(rho, Lw, rs):
    area = 2 * Lw * Lw; N = rs.poisson(rho * area)
    u = rs.uniform(-Lw, Lw, (N, 2)); X = np.stack([(u[:, 0] + u[:, 1]) / 2, (u[:, 0] - u[:, 1]) / 2], 1)
    return X[np.abs(X[:, 0]) + np.abs(X[:, 1]) <= Lw]

if __name__ == '__main__':
    rho, Lw = 4.0, 14.0
    M0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    dts = (1.0, 2.0, 4.0, 6.0, 8.0)
    worlds = [('золото', CH.sym_world(rho, Lw).X)] + [(f'Пуассон {i}', poisson_world(rho, Lw, np.random.default_rng(i))) for i in range(2)]
    for name, X in worlds:
        W = Polar(X, m0=M0)
        srcs = [k for k in range(W.n) if abs(X[k, 0] + 9) < 0.6 and abs(X[k, 1]) < 1.5]
        rows = {eta: [] for eta in (0.3, 0.8, 1.5)}
        for s in srcs[:40]:
            lo = np.array(W.eout[s])
            for eta in rows:
                if len(lo) == 0: continue
                e0 = lo[np.argmin(np.abs(W.rap[lo] - eta))]
                if abs(W.rap[e0] - eta) > 0.25: continue
                rows[eta].append((math.tanh(W.rap[e0]), measure(W, e0, dts)))
        print(f'{name}: событий {W.n}, звеньев на событие {W.E / W.n:.1f}')
        for eta, rr in rows.items():
            if not rr: continue
            v0 = np.mean([r[0] for r in rr]); M = np.nanmean([r[1] for r in rr], 0)
            print(f'   старт η ≈ {eta} (скорость ноги {v0:.2f}, стартов {len(rr)}): средняя скорость по срезам '
                  f'{" ".join(f"{v:+.2f}" for v in M[:, 0])}; в мире {" ".join(f"{p:.2f}" for p in M[:, 1])}; разброс скорости {" ".join(f"{s:.2f}" for s in M[:, 2])}')
        sys.stdout.flush()
