#!/usr/bin/env python3
"""ЗАМКНУТАЯ СИСТЕМА С ТОЧНЫМИ СИММЕТРИЯМИ КРИСТАЛЛА («полярное событие»).

Две находки про closed2.py:
1) «поворот = i» там — чистая калибровка: фаза ноги входит как ph на выходе события-начала и ph* на входе
   события-конца и сокращается; amp_с_i(e) = ph(e)·amp_без_i(e) до 6e−17. Динамика вещественна; i — только в том,
   как детектор складывает входящие ноги.
2) closed2 ломает зеркало P на ~100 %: пары «вход k → выход k» в матрице Гровера назначены по номеру в порядке
   быстрот (при числе входов ≠ выходов это не переходит в себя при x → −x), а паросочетание по быстроте даёт
   тысячи ничьих (звенья одного направления разной длины).

Новое правило события — без назначенных пар: ядро K_ji = exp(−(d_out_j · d_in_i)/m₀), где · — произведение
Минковского векторов ног (не меняется ни при бусте, ни при зеркале), и событие = ближайшая к K изометрия
(полярное разложение V = A·Bᵀ из K = A Σ Bᵀ). Перестановка ног переставляет V так же — ни номеров, ни ничьих.
Если входов больше, чем выходов, V — коизометрия, недостающая норма уходит в резервуар тени: |v|² − |Vv|².
Скрытые ноги — только P-замкнутое множество. Впрыск — симметричная мода выходов; захват — ⟨s|v_in⟩.
Динамика вещественна (комплексность — от поля, как в тесте Renou)."""
import sys, math, numpy as np
from scipy.optimize import linear_sum_assignment
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a

def mink(a, b): return a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]

class ClosedSym:
    def __init__(self, rho=4.0, Lw=24.0, m0=2.0, psym=True, kernel='exp', tol=1e-9, match=None):
        import os; match = match or os.environ.get('MATCH', 'глобальное')
        self.tol = tol
        w = CH.sym_world(rho, Lw); R = CH.build_order(w, 'proj'); Lk = CH.links_of(R); LK = CH.prep(w, Lk, 1)
        X = w.X; n = w.n; self.w, self.R, self.X, self.n = w, R, X, n
        I, J = LK['I'], LK['J']
        ins, outs = np.bincount(J, minlength=n), np.bincount(I, minlength=n)
        exo = np.maximum(ins - outs, 0); exi = np.maximum(outs - ins, 0)
        U = np.repeat(np.arange(n), exo); V = np.repeat(np.arange(n), exi)
        if match == 'глобальное':          # минимум суммы теневых расстояний по всему миру (как в closed2)
            C = np.sqrt(((w.W[U][:, None, :] - w.W[V][None, :, :]) ** 2).sum(-1))
            C = np.where(R[np.ix_(U, V)], C, 1e6); r, c = linear_sum_assignment(C); ok = C[r, c] < 1e6
            hid = list(zip(U[r[ok]].tolist(), V[c[ok]].tolist()))
        elif match == 'нет':               # скрытых ног нет: всё лишнее — в резервуар тени
            hid = []
        else:                              # 'взаимное': пара — взаимно ближайшие в тени (жадно по расстоянию), без оптимизации по миру
            Ue, Ve = np.nonzero(exo)[0], np.nonzero(exi)[0]
            C = np.sqrt(((w.W[Ue][:, None, :] - w.W[Ve][None, :, :]) ** 2).sum(-1)); ca = R[np.ix_(Ue, Ve)]
            iu, iv = np.nonzero(ca); o = np.argsort(np.round(C[iu, iv], 12), kind='stable')
            eo, ei = exo.copy(), exi.copy(); hid = []
            for a, b in zip(Ue[iu[o]], Ve[iv[o]]):
                m = min(eo[a], ei[b])
                if m > 0: hid += [(int(a), int(b))] * int(m); eo[a] -= m; ei[b] -= m
        self.Pimg = np.array([CH.img(w, k, CH.P, CH.P) for k in range(n)])
        if psym:
            from collections import Counter
            cnt = Counter(hid); hid = []
            for (a, b), c in cnt.items(): hid += [(a, b)] * min(c, cnt.get((int(self.Pimg[a]), int(self.Pimg[b])), 0))
        self.n_hidden, self.n_excess = len(hid), len(U)
        self.edges = [(int(a), int(b), 0) for a, b in zip(I, J)] + [(a, b, 1) for a, b in hid]
        self.E = len(self.edges)
        self.d = np.array([X[b] - X[a] for a, b, _ in self.edges])
        ein = [[] for _ in range(n)]; eout = [[] for _ in range(n)]
        for e, (a, b, k) in enumerate(self.edges): eout[a].append(e); ein[b].append(e)
        self.order = list(np.argsort(X[:, 0], kind='stable'))
        lk = np.array([mink(self.d[e], self.d[e]) for e in range(len(I))])
        self.m0 = m0 or float(np.median(lk))
        self.f = np.ones(self.E)           # фазы поля на ногах (set_field)
        self.legs = []; self.minsv = np.inf; self.ndeg = 0
        for k in range(n):
            li, lo = np.array(ein[k], int), np.array(eout[k], int)
            if len(li) and len(lo):
                M = mink(self.d[lo][:, None, :], self.d[li][None, :, :])
                K = np.exp(-M / self.m0) if kernel == 'exp' else 1.0 / M
                A, s, Bt = np.linalg.svd(K, full_matrices=False); keep = s > self.tol * s.max()
                Vm = A[:, keep] @ Bt[keep]                       # частичная изометрия: вырожденное (одинаковые ноги) — в резервуар
                self.minsv = min(self.minsv, s[keep].min() / s.max()); self.ndeg += int((~keep).sum())
            else: Vm = np.zeros((len(lo), len(li)))
            self.legs.append((li, lo, Vm))

    def run(self, inj, block=None, record=None, plate=None, slices=None):
        """inj: {событие: амплитуды по столбцам} — излучается в симметричную моду выходов.
        record: события, где снимается ⟨s|v_in⟩. slices: времена, на которых считать норму «мир + тень»."""
        ncol = len(next(iter(inj.values())))
        amp = np.zeros((self.E, ncol), complex); ab = {}; rec = {}; resk = np.zeros((self.n, ncol))
        for k in self.order:
            li, lo, Vm = self.legs[k]
            vin = amp[li] if len(li) else np.zeros((0, ncol), complex)
            if record is not None and k in record: rec[k] = vin.sum(0) / math.sqrt(max(len(li), 1))
            vout = Vm @ vin if len(li) and len(lo) else np.zeros((len(lo), ncol), complex)
            resk[k] = (np.abs(vin) ** 2).sum(0) - (np.abs(vout) ** 2).sum(0)
            if k in inj and len(lo): vout = vout + np.outer(np.ones(len(lo)) / math.sqrt(len(lo)), inj[k])
            for j, e in enumerate(lo):
                a = vout[j]
                if plate is not None and e in plate: a = a * plate[e]
                if block is not None and e in block: ab[e] = a; continue
                amp[e] = a * self.f[e]
        out = (ab, resk, amp, rec)
        if slices is not None:
            X = self.X; ea = np.array([e[0] for e in self.edges]); eb = np.array([e[1] for e in self.edges])
            nm = []
            for t in slices:
                cr = (X[ea, 0] < t) & (X[eb, 0] >= t)
                nm.append((np.abs(amp[cr]) ** 2).sum(0) + resk[X[:, 0] < t].sum(0) + sum(np.abs(ab[e]) ** 2 for e in ab if X[eb[e], 0] < t or X[ea[e], 0] < t) * 0)
            out = out + (np.array(nm),)
        return out

    def set_field(self, B):
        """однородное поле закрученного среза: фаза ноги a → b = (B/2)(T_a x_b − x_a T_b), B = qF (fieldtwist.py)"""
        X = self.X; a = np.array([e[0] for e in self.edges]); b = np.array([e[1] for e in self.edges])
        self.f = np.exp(1j * B / 2 * (X[a, 0] * X[b, 1] - X[a, 1] * X[b, 0])) if B else np.ones(self.E)

    def backward(self, rec_events):
        """β[e, c]: амплитуда с ноги e до ⟨s|v_in⟩ в rec_events[c]; emit[k, c] — от симметричного излучения в k."""
        col = {k: i for i, k in enumerate(rec_events)}; nc = len(rec_events)
        dt = complex if np.iscomplexobj(self.f) else float
        beta = np.zeros((self.E, nc), dt); emit = np.zeros((self.n, nc), dt)
        for k in reversed(self.order):
            li, lo, Vm = self.legs[k]
            bout = beta[lo] if len(lo) else np.zeros((0, nc), dt)
            if len(lo): emit[k] = bout.sum(0) / math.sqrt(len(lo))
            if len(li):
                bin_ = Vm.T @ bout if len(lo) else np.zeros((len(li), nc), dt)
                if k in col: bin_[:, col[k]] += 1 / math.sqrt(len(li))
                beta[li] = bin_ * self.f[li][:, None]
        return beta, emit, col

if __name__ == '__main__':
    S = ClosedSym(); X = S.X; n = S.n; Pimg = S.Pimg
    print(f'событий {n}; скрытых ног {S.n_hidden} из лишних {S.n_excess}; m₀ = {S.m0:.3f}; мин. отношение сингулярных чисел ядра {S.minsv:.1e}; вырожденных направлений {S.ndeg}')
    eid = {(a, b, kd): e for e, (a, b, kd) in enumerate(S.edges)}
    Pe = np.array([eid.get((Pimg[a], Pimg[b], kd), -1) for (a, b, kd) in S.edges])
    # буст-образ (g) для проверки ковариантности
    Gimg = np.array([CH.img(S.w, k, CH.G, CH.Gs) for k in range(n)])
    Ge = np.array([eid.get((Gimg[a], Gimg[b], kd), -1) if Gimg[a] >= 0 and Gimg[b] >= 0 else -1 for (a, b, kd) in S.edges])
    rng = np.random.default_rng(0)
    srcs = rng.choice([k for k in range(n) if abs(X[k, 0]) < 6 and abs(X[k, 1]) < 3], 5, replace=False)
    for s in srcs:
        s = int(s)
        _, rk, a1, _, nm = S.run({s: np.array([1.0 + 0j])}, slices=[X[s, 0] + 0.5, 0.0, 5.0, 10.0])
        _, _, a2, _ = S.run({int(Pimg[s]): np.array([1.0 + 0j])})
        m = Pe >= 0; aP = np.abs(a1[m, 0] - a2[Pe[m], 0]).max() / np.abs(a1[:, 0]).max()
        msg = f'источник {np.round(X[s], 2)}: нарушение P {aP:.1e}'
        if Gimg[s] >= 0:
            _, _, a3, _ = S.run({int(Gimg[s]): np.array([1.0 + 0j])})
            # сравниваем только ноги далеко от края мира (образ существует и у всех ног обоих событий)
            m3 = (Ge >= 0) & (np.abs(X[[e[0] for e in S.edges], 0]) < 8)
            msg += f'; нарушение буста g {np.abs(a1[m3, 0] - a3[Ge[m3], 0]).max() / np.abs(a1[:, 0]).max():.1e}'
        print(msg + f'; норма мир + тень на срезах {np.round(nm[:, 0], 9)}'); sys.stdout.flush()
