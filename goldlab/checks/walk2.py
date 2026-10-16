"""ДИРАК НА КРИСТАЛЛЕ, шаг 2: унитарное блуждание по световым нитям. В каждом событии — монета
[[cos θ, −i sin θ], [−i sin θ, cos θ]], θ = m·Δt̄ (Δt̄ — средний входящий шаг). Проверки: (1) норма сохраняется (вход = на срезе + ушло
за край); (2) спектр: удар волной e^{ikx} в обе компоненты — частоты ±E(k): ждём две ветви ±√(m² + c²k²) (частица и античастица);
(3) скорость безмассовой частицы c (нити выбраны в системе покоя кристалла — проверяем, насколько она близка к 1)."""
import sys, os, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from scipy.optimize import linear_sum_assignment
from goldlab import worlds as Wd, causal as Cs, observe as Ob
def threads(X, side, cap, cost='dt', big=1e6):
    n = len(X); Cr = Cs.causal(X); la, lb = np.nonzero(Cr); dx = X[lb, 1] - X[la, 1]; dt = X[lb, 0] - X[la, 0]
    ok = (dx * side > 1e-9) & (dt < cap); C = np.full((n, n), big)
    C[la[ok], lb[ok]] = dt[ok] if cost == 'dt' else np.sqrt(np.maximum(dt[ok] ** 2 - dx[ok] ** 2, 0)) + 1e-3 * dt[ok]
    r, c = linear_sum_assignment(C); nxt = -np.ones(n, int); good = C[r, c] < big; nxt[r[good]] = c[good]; return nxt
class Walk:
    def __init__(self, w, cap=1.6, cost='dt'):
        self.w = w; X = w.X; n = w.n; self.R = threads(X, +1, cap, cost); self.L = threads(X, -1, cap, cost)
        self.pR = -np.ones(n, int); self.pR[self.R[self.R >= 0]] = np.nonzero(self.R >= 0)[0]
        self.pL = -np.ones(n, int); self.pL[self.L[self.L >= 0]] = np.nonzero(self.L >= 0)[0]
        t = X[:, 0]; self.dtR = np.where(self.pR >= 0, t - t[np.maximum(self.pR, 0)], np.nan); self.dtL = np.where(self.pL >= 0, t - t[np.maximum(self.pL, 0)], np.nan)
        sR = X[self.R[self.R >= 0]] - X[self.R >= 0]; sL = X[self.L[self.L >= 0]] - X[self.L >= 0]
        self.c = 0.5 * (np.mean(sR[:, 1] / sR[:, 0]) - np.mean(sL[:, 1] / sL[:, 0]))
    def run(self, m, fR, fL):
        """fR, fL — вклады источника во входы событий. Возвращает входы (aR, aL) каждого события и норму, ушедшую за край."""
        w = self.w; t = w.t; n = w.n; aR = fR.astype(complex).copy(); aL = fL.astype(complex).copy(); leak = 0.0
        dtb = np.nanmean(np.c_[self.dtR, self.dtL], axis=1); dtb = np.where(np.isnan(dtb), np.nanmedian(dtb), dtb)
        outR = np.zeros(n, complex); outL = np.zeros(n, complex)
        for z in np.argsort(t, kind='stable'):
            th = m * dtb[z]; c, s = np.cos(th), np.sin(th)
            bR = c * aR[z] - 1j * s * aL[z]; bL = -1j * s * aR[z] + c * aL[z]; outR[z], outL[z] = bR, bL
            if self.R[z] >= 0: aR[self.R[z]] += bR
            else: leak += abs(bR) ** 2
            if self.L[z] >= 0: aL[self.L[z]] += bL
            else: leak += abs(bL) ** 2
        return aR, aL, outR, outL, leak
if __name__ == '__main__':
    L = float(os.environ.get('L', 16)); w = Wd.gold11(4.0, L); t, x = w.t, w.x
    for cost in ('dt', 'tau'):
        W = Walk(w, cap=float(os.environ.get('CAP', 1.6)), cost=cost); m = 1.0
        ts = -L + 4; src = np.abs(t - ts) < 0.45; f = np.where(src, np.exp(-x * x / 8), 0)
        aR, aL, oR, oL, leak = W.run(m, f, 0 * f)
        inj = (np.abs(f) ** 2).sum(); T = 0.0; cut = 0.0
        for z in range(w.n):                                   # норма на срезе t = T: рёбра, пересекающие T
            for nx, o in ((W.R[z], oR[z]), (W.L[z], oL[z])):
                if t[z] < T <= (t[nx] if nx >= 0 else -1e9): cut += abs(o) ** 2
        leak_before = 0.0
        # (грубо) утечка до T — пересчёт с остановкой
        print(f'нити по {"Δt" if cost == "dt" else "собственному времени"}: скорость безмассовой c = {W.c:.3f}; полная норма: вход {inj:.4f}, ушло за край (всего) {leak:.4f}', flush=True)
        # спектр: удар волной e^{ikx} (обе компоненты), чтение на поздних событиях
        om = np.linspace(-3.5, 3.5, 2801)
        for k in (0.0, 0.5, 1.0):
            g = np.exp(-x * x / 50) * np.exp(1j * k * x); fR = np.where(src, g, 0); aR, aL, _, _, _ = W.run(m, fR, fR)
            sel = (t > ts + 1.5) & (t < L - 4) & (np.abs(x) < 8); kk = np.round(t[sel], 6); tu, inv = np.unique(kk, return_inverse=True)
            s = np.zeros(len(tu), complex); np.add.at(s, inv, (np.conj(g) * (aR + aL))[sel])
            S = Ob.spectrum(tu, s, om); S /= S.max(); pk = Ob.peaks(om, S, 0.08)
            print(f'   k = {k}: пики ω: ' + ' '.join(f'{e:+.3f}({q:.2f})' for e, q in pk) + f'   (ожидание ±√(m² + c²k²) = ±{np.sqrt(m * m + (W.c * k) ** 2):.3f})', flush=True)
