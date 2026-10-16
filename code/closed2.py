#!/usr/bin/env python3
"""ЗАМКНУТАЯ СИСТЕМА «МИР + ТЕНЬ» БЕЗ ОТРЕЗА, С «ПОВОРОТ = i».

* Все звенья порядка (никакого Δt — правило одинаково в любой системе отсчёта кристалла).
* Лишние ноги событий спариваются причинно (X ≺ Y) по близости теней; что не спарилось — нога в «резервуар
  тени»: недостающий вход приходит из резервуара в известном (пустом) состоянии, лишний выход уходит в резервуар
  (как новые степени свободы у Котлера–Строминджера). Резервуар — часть тени, норма считается вместе с ним.
* Событие — унитарная матрица D × D: U = diag(i^{ℓ_out}) · G · diag(i^{−ℓ_in}), G = (2/D)·1 − I (Гровер),
  ℓ — метка киральности ноги (0/1; «и туда, и туда» и скрытые — 0). Переход с «лево» на «право» получает ±i,
  вещественная структура сохраняется (G вещественна).
Ноги упорядочены по быстроте (скрытые и резервуар — в конце); пара «вход k → выход k» получает −1 + 2/D
(отражение по Гроверу), остальные 2/D."""
import sys, math, numpy as np
from scipy.optimize import linear_sum_assignment
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a

class Closed:
    def __init__(self, rho=4.0, Lw=24.0, turn=True, match_kind='тень'):
        w = CH.sym_world(rho, Lw); R = CH.build_order(w, 'proj'); Lk = CH.links_of(R); LK = CH.prep(w, Lk, 1)
        X = w.X; n = w.n; self.w, self.R, self.X, self.n = w, R, X, n
        I, J, lab = LK['I'], LK['J'], LK['lab']
        ins, outs = np.bincount(J, minlength=n), np.bincount(I, minlength=n)
        exo = np.maximum(ins - outs, 0); exi = np.maximum(outs - ins, 0)
        U = np.repeat(np.arange(n), exo); V = np.repeat(np.arange(n), exi)
        caus = R[np.ix_(U, V)]
        C = np.sqrt(((w.W[U][:, None, :] - w.W[V][None, :, :]) ** 2).sum(-1)) if match_kind == 'тень' else \
            np.sqrt(np.maximum((X[V][None, :, 0] - X[U][:, None, 0]) ** 2 - (X[V][None, :, 1] - X[U][:, None, 1]) ** 2, 0))
        C = np.where(caus, C, 1e6); r, c = linear_sum_assignment(C); ok = C[r, c] < 1e6
        ha, hb = U[r[ok]], V[c[ok]]
        self.n_hidden, self.n_excess = len(ha), len(U)
        # ноги: (откуда, куда, вид, метка); вид 0 — мир, 1 — скрытая
        self.edges = [(int(a), int(b), 0, int(l)) for a, b, l in zip(I, J, lab)] + [(int(a), int(b), 1, 2) for a, b in zip(ha, hb)]
        E = len(self.edges); self.E = E
        ein = [[] for _ in range(n)]; eout = [[] for _ in range(n)]
        for e, (a, b, k, l) in enumerate(self.edges): eout[a].append(e); ein[b].append(e)
        def rap(e):
            a, b, k, l = self.edges[e]
            if k: return 10.0 + e * 1e-7
            d = X[b] - X[a]; return math.atanh(d[1] / d[0])
        self.order = list(np.argsort(X[:, 0], kind='stable'))
        self.legs = []
        for k in range(n):
            li = sorted(ein[k], key=rap); lo = sorted(eout[k], key=rap); D = max(len(li), len(lo), 1)
            Gm = 2.0 / D * np.ones((D, D)) - np.eye(D)
            lin = np.array([self.edges[e][3] for e in li] + [2] * (D - len(li))); lout = np.array([self.edges[e][3] for e in lo] + [2] * (D - len(lo)))
            ph = lambda l: np.where(l == 1, 1j, 1.0) if turn else np.ones(len(l))
            Um = np.diag(ph(lout)) @ Gm @ np.diag(np.conj(ph(lin)))
            self.legs.append((np.array(li, int), np.array(lo, int), Um))
        self.maxdev = max(np.abs(Um.conj().T @ Um - np.eye(len(Um))).max() for _, _, Um in self.legs)

    def run(self, inj, block=None, plate=None, ncol=None):
        """inj: {событие: вектор амплитуд по столбцам (несколько источников сразу)}; block: {нога: метка поглотителя};
        plate: {нога: фаза}. Возвращает: поглощённое по ногам-поглотителям, ушедшее в резервуар, амплитуды на ногах."""
        ncol = ncol or len(next(iter(inj.values())))
        amp = np.zeros((self.E, ncol), complex); res = np.zeros(ncol); absorbed = {}
        for k in self.order:
            li, lo, Um = self.legs[k]; D = len(Um)
            vin = np.zeros((D, ncol), complex); vin[:len(li)] = amp[li]
            if k in inj: vin[len(li) if len(li) < D else 0] += inj[k]
            vout = Um @ vin
            res += (np.abs(vout[len(lo):]) ** 2).sum(0)
            for idx, e in enumerate(lo):
                a = vout[idx]
                if plate is not None and e in plate: a = a * plate[e]
                if block is not None and e in block: absorbed[e] = a; continue
                amp[e] = a
        return absorbed, res, amp

if __name__ == '__main__':
    for turn in (True, False):
        S = Closed(turn=turn)
        print(f'«поворот = i»: {turn}; событий {S.n}, ног-звеньев {S.E - S.n_hidden}, скрытых {S.n_hidden} из лишних {S.n_excess} '
              f'(остальные — в резервуар тени); унитарность матриц событий до {S.maxdev:.1e}')
        X = S.X
        src = int(np.argmin(((X - np.array([-12.0, 0.0])) ** 2).sum(1)))
        absorbed, res, amp = S.run({src: np.array([1.0 + 0j])})
        tot = res.sum() + (np.abs(amp) ** 2).sum()
        print(f'  источник {np.round(X[src], 1)}: ушло в резервуар {res[0]:.4f}, осталось на ногах {(np.abs(amp) ** 2).sum():.4f}, сумма {tot:.6f}')
        for t in (-8.0, -4.0, 0.0, 4.0, 8.0):
            cw = [e for e, (a, b, kk, l) in enumerate(S.edges) if X[a, 0] < t <= X[b, 0] and kk == 0]
            ch = [e for e, (a, b, kk, l) in enumerate(S.edges) if X[a, 0] < t <= X[b, 0] and kk == 1]
            print(f'    срез t = {t:+.0f}: в мире {np.sum(np.abs(amp[cw]) ** 2):.4f}, в скрытых ногах {np.sum(np.abs(amp[ch]) ** 2):.4f}')
        sys.stdout.flush()
