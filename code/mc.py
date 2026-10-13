#!/usr/bin/env python3
"""Вес на расстановках: Монте-Карло по exp(-beta*S).

ПОВОД. weight-doc.md: объект находится (вмятина до 13σ, следует за
собой), но массы нет, и оба оставшихся провала сходятся в одну точку —
нужен вес на РАССТАНОВКАХ exp(-S). Первый пункт no-dynamics-doc.md и
после сегодняшнего единственный.

ПРЕИМУЩЕСТВО ЭТОЙ ВЕРСИИ. Известная беда суммирования по причинным
множествам — энтропия трёхслойных порядков (Клейтман–Ротшильд): их
астрономически больше манифолдоподобных. Здесь выборка идёт по
ПОЛОЖЕНИЯМ событий в области Минковского, поэтому манифолдоподобный
сектор не покидается по построению и трёхслойный порядок недостижим.

ГЛАВНЫЙ ВОПРОС. Притягивает ли exp(-beta*S) фон к локальному объекту?
Если да — это источник, то есть тяготение из веса, и тогда у изгиба
появится цена. Если нет — вес расстановок веществом не делает.

КОНТРОЛИ.
  beta = 0 — равномерная выборка, задаёт эталон плотности (все
             геометрические поправки формы куба делятся сами);
  рассыпанный объект той же массы вместо трубки;
  --check — сверка инкрементного обновления btw с полным пересчётом.

ОГРАНИЧЕНИЕ, КОТОРОЕ НАДО ЗНАТЬ. Дальнобойность веса относительно
размера мира зависит ТОЛЬКО от числа событий: reach/size ~ (38/N)^(1/4).
При N = 600 это 0.5, то есть промежуточный масштаб вполовину мира. Чтобы
получить 0.3, нужно ~4700 событий, а это Метрополису недоступно.

  python3 mc.py --check
  python3 mc.py --sweeps 150
"""
import argparse
import math
import time

import numpy as np

from weight import w_of

T_MAX = 4.0


def box(N, L, rs):
    X = np.empty((N, 4))
    X[:, 0] = rs.uniform(0, T_MAX, N)
    X[:, 1:] = rs.uniform(-L, L, (N, 3))
    return X


def tube(M, rad, rs):
    v = rs.normal(size=(M, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = rs.uniform(0, T_MAX, M)
    X[:, 1:] = v * r[:, None]
    return X


def bend(X, delta):
    Y = X.copy()
    late = Y[:, 0] > T_MAX / 2
    Y[late, 1] += delta * (Y[late, 0] - T_MAX / 2)
    return Y


class World:
    """Расстановка с действием, поддерживаемым инкрементно."""

    def __init__(self, X, eps):
        self.X = X.copy()
        n = len(X)
        self.cap = n
        self.wtab = w_of(np.arange(n + 1), eps)
        dt = X[:, None, 0] - X[None, :, 0]
        dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=-1)
        R = dt > dx
        np.fill_diagonal(R, False)
        self.R = np.asarray(R, bool)
        self.A = self.R.astype(np.float32)
        self.btw = np.rint(self.A @ self.A).astype(np.int32)
        self.S = self._S()
        self.cnt = int(self.R.sum())        # число связанных пар

    def _S(self):
        return float(self.wtab[np.clip(self.btw, 0, self.cap)][self.R].sum())

    def full_btw(self):
        return np.rint(self.A @ self.A).astype(np.int32)

    def _rowcol(self, i, xi):
        dt = xi[0] - self.X[:, 0]
        dx = np.linalg.norm(xi[1:] - self.X[:, 1:], axis=-1)
        u = dt > dx                      # R[i,:]: эти раньше i
        v = (-dt) > dx                   # R[:,i]: i раньше этих
        u[i] = v[i] = False
        return u, v

    def move(self, i, xi):
        """Применить перенос события i; вернуть данные для отката."""
        u0 = self.R[i].copy()
        v0 = self.R[:, i].copy()
        brow = self.btw[i].copy()
        bcol = self.btw[:, i].copy()
        x0 = self.X[i].copy()
        S0 = self.S
        c0 = self.cnt

        u1, v1 = self._rowcol(i, xi)
        self.btw += np.outer(v1.astype(np.int32), u1.astype(np.int32))
        self.btw -= np.outer(v0.astype(np.int32), u0.astype(np.int32))
        self.X[i] = xi
        self.R[i] = u1
        self.R[:, i] = v1
        self.A[i] = u1
        self.A[:, i] = v1
        self.btw[i] = np.rint(u1.astype(np.float32) @ self.A)
        self.btw[:, i] = np.rint(self.A @ v1.astype(np.float32))
        self.S = self._S()
        self.cnt += (int(u1.sum()) + int(v1.sum())
                     - int(u0.sum()) - int(v0.sum()))
        return (i, x0, u0, v0, u1, v1, brow, bcol, S0, c0)

    def undo(self, st):
        i, x0, u0, v0, u1, v1, brow, bcol, S0, c0 = st
        self.btw -= np.outer(v1.astype(np.int32), u1.astype(np.int32))
        self.btw += np.outer(v0.astype(np.int32), u0.astype(np.int32))
        self.X[i] = x0
        self.R[i] = u0
        self.R[:, i] = v0
        self.A[i] = u0
        self.A[:, i] = v0
        self.btw[i] = brow
        self.btw[:, i] = bcol
        self.S = S0
        self.cnt = c0


def chain(obj, N, L, beta, eps, sweeps, rs, step=0.4, check=False):
    M = len(obj)
    w = World(np.vstack([obj, box(N, L, rs)]), eps)
    trace, acc, prof, pc = [w.S], 0, np.zeros(6), 0
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.9, 1.3, 99.0])
    for sw in range(sweeps):
        for _ in range(N):
            i = M + int(rs.integers(N))
            xi = w.X[i].copy()
            xi[0] = min(max(xi[0] + step * rs.normal(), 0.0), T_MAX)
            xi[1:] = np.clip(xi[1:] + step * rs.normal(size=3), -L, L)
            st = w.move(i, xi)
            if check:
                assert np.array_equal(w.btw, w.full_btw()), 'btw расходится'
            d = w.S - st[8]
            if d <= 0 or rs.random() < math.exp(-beta * d):
                acc += 1
            else:
                w.undo(st)
        trace.append(w.S)
        if sw >= sweeps // 2:
            r = np.linalg.norm(w.X[M:, 1:], axis=1)
            prof += np.bincount(np.digitize(r, edges) - 1, minlength=6)[:6]
            pc += 1
    return w, np.array(trace), acc / (sweeps * N), prof / max(pc, 1)


def main():
    ap = argparse.ArgumentParser(description='Монте-Карло по exp(-beta S)')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--m', type=int, default=100)
    ap.add_argument('--l', type=float, default=1.0)
    ap.add_argument('--rad', type=float, default=0.05)
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--sweeps', type=int, default=150)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    if a.check:
        rs = np.random.default_rng(0)
        chain(tube(20, a.rad, rs), 40, a.l, 1.0, a.eps, 2, rs, check=True)
        print('инкрементное обновление btw совпадает с полным пересчётом')
        return

    t0 = time.time()
    ntot = a.n + a.m
    print(f'фон {a.n} подвижных, объект {a.m} закреплённых, L={a.l}, '
          f'eps={a.eps}')
    print(f'reach/size ~ {(38.0/ntot)**0.25:.2f} при N={ntot}')
    print(f'развёрток {a.sweeps}, засевов {a.seeds}')
    print()
    print('ПЛОТНОСТЬ ФОНА ПО |x|, отнесённая к beta=0 того же мира')
    print(f"{'мир':>10}{'beta':>6}{'принят':>8}"
          f"{'<0.2':>8}{'0.2-0.4':>9}{'0.4-0.6':>9}{'0.6-0.9':>9}"
          f"{'0.9-1.3':>9}{'дрейф <S>':>11}{'<S>':>10}")

    for lbl, mk in (('трубка', True), ('контроль', False)):
        ref = None
        for beta in (0.0, 1.0, 3.0):
            ps, ac, dr, sv = [], [], [], []
            for t in range(a.seeds):
                rs = np.random.default_rng(1100 + 17 * t)
                obj = tube(a.m, a.rad, rs) if mk else box(a.m, a.l, rs)
                _, tr, r, p = chain(obj, a.n, a.l, beta, a.eps,
                                    a.sweeps, rs)
                ps.append(p)
                ac.append(r)
                h = len(tr) // 2
                dr.append(tr[h:].mean() - tr[:h].mean())
                sv.append(tr[h:].mean())
            P = np.array(ps).mean(0)
            if ref is None:
                ref = P.copy()
            Q = P / np.maximum(ref, 1e-9)
            print(f'{lbl:>10}{beta:>6.1f}{np.mean(ac):>8.2f}'
                  + ''.join(f'{v:>9.3f}' for v in Q[:5])
                  + f'{np.mean(dr):>11.2f}{np.mean(sv):>10.2f}',
                  flush=True)
        print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
