#!/usr/bin/env python3
"""ПОЛЕ ИЗ ИЗОГНУТОГО СРЕЗА ВМЕСТО РУЧНЫХ ПОТОКОВ.

Закрученный срез (twist.py: h₁ = κ(T²+x²)/2, h₂ = κTx) даёт связность нормального расслоения
A = κ²(x, −T) (проверено численно до 1e−4), т.е. однородное поле F = −2κ². Частица с «зарядом» q
получает на звене I → J фазу q∫A·dl = (qF/2)·(T_I x_J − x_I T_J); вокруг петли — q·(поток).
Параметр опыта — B = qF (поле, умноженное на заряд).

Проверяем на кристалле ρ = 4 с правилом «поворот = i»:
  1) потери делителей (теорема чётности больше не точна);
  2) насколько ломается вещественность (класс амплитуд);
  3) Renou БЕЗ пластинок: настройки Алисы и Чарли — выбор одного из делителей кристалла (разные
     установки охватывают разный поток) или прямое наблюдение пути (Z). Боб — идеальный Белл.
     Берём только делители с потерями ниже порога.
"""
import sys, math, itertools, os, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from renou import BELL, local_search, phip, Z
sys.argv = _a

def twist_check():
    from twist import make, connection
    kap = 0.05; S, tang, frame = make(kap, True); err = 0
    for T, x in ((1.0, 2.0), (-3.0, 4.0), (5.0, -2.0)):
        A = connection(frame, T, x); err = max(err, np.abs(A - kap * kap * np.array([x, -T])).max() / (kap * kap * 5))
    return err

def build(rho=4.0, K=2, LABK=1):
    w = CH.sym_world(rho, 24.0); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), LABK)
    return w, R, LK

def collect(w, R, LK, B, K=2):
    X = w.X; n = w.n; I, J = LK['I'], LK['J']
    fld = np.exp(1j * B / 2 * (X[I, 0] * X[J, 1] - X[I, 1] * X[J, 0]))
    H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
    Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)
    src = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
    PS = {(s, m): CH.chiral(w, LK, R, s, m, field=fld) for s in src for m in (0, 1)}
    Ms, Ts, mixed = [], [], 0.0
    for v in PS.values():
        a = np.abs(v).max(); mixed = max(mixed, (np.minimum(np.abs(v.real), np.abs(v.imag)) > 1e-9 * a).mean())
    for A1 in [k for k in range(n) if 2.0 <= Xa[k, 0] <= 3.2 and 0.3 <= Xa[k, 1] <= 2.0]:
        A2 = CH.img(w, A1, H, Hs)
        if A2 < 0: continue
        B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
        for m0 in (0, 1):
            za = np.array([[PS[s, m][A1, m0], PS[s, m][A2, 1 - m0]] for s in src for m in (0, 1)])
            zb = np.array([[PS[s, m][B1, 1 - m0], PS[s, m][B2, m0]] for s in src for m in (0, 1)])
            Ms.append(za.T @ zb)
            ZA = {(x, mm): CH.chiral(w, LK, R, x, mm, field=fld) for x, mm in ((A1, m0), (A2, 1 - m0))}
            for m1 in (0, 1):
                for p in range(n):
                    if not (4.0 <= Xa[p, 0] <= 6.5 and Xa[p, 1] > 0.05 and R[A1, p] and R[A2, p]): continue
                    q = CH.img(w, p, H, Hs)
                    if q < 0: continue
                    T = np.array([[ZA[A1, m0][p, m1], ZA[A2, 1 - m0][p, m1]], [ZA[A1, m0][q, 1 - m1], ZA[A2, 1 - m0][q, 1 - m1]]])
                    if (np.abs(T) ** 2).sum() > 1e-24: Ts.append(T)
    return np.array(Ms), np.array(Ts), mixed

D = np.diag([1, -1j]); Di = np.diag([1, 1j])
if __name__ == '__main__':
    print(f'связность закрученного среза = κ²(x, −T): отклонение {twist_check():.1e}')
    w, R, LK = build()
    rows = []
    for B in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3):
        Ms, Ts, mixed = collect(w, R, LK, B)
        u = np.array([2 * abs(np.linalg.det(T)) / (np.abs(T) ** 2).sum() for T in Ts]); loss = 1 - u
        Tf = [D @ T @ Di for T in Ts]; Mf = [D @ M @ D.T for M in Ms]
        # «комплексность» наблюдаемой: y-компонента вектора Блоха (в базисе f пустой кристалл даёт y = 0)
        def obs(T):
            U, _, Vh = np.linalg.svd(T); V = U @ Vh; return V.conj().T @ Z @ V
        Y = np.array([[0, -1j], [1j, 0]])
        ycomp = np.array([abs(np.trace(obs(T) @ Y).real) / 2 for T in Tf])
        conc = np.array([2 * abs(np.linalg.det(M)) / (np.abs(M) ** 2).sum() for M in Mf])
        res = {}
        for tol in (1e-3, 1e-2):
            good = [T for T, l in zip(Tf, loss) if l < tol]
            if len(good) < 2: res[tol] = (None, 0); continue
            # разные установки — по одному на ячейку (угол поворота, y)
            keys = {}
            for T in good:
                o = obs(T); k = (round(np.trace(o @ np.array([[0, 1], [1, 0]])).real / 2 / 0.05), round(np.trace(o @ Y).real / 2 / 0.05), round(np.trace(o @ Z).real / 2 / 0.05))
                keys.setdefault(k, T)
            obs_set = [Z] + [obs(T) for T in keys.values()]
            best = 0
            for i in np.argsort(-conc)[:3]:
                for pm in itertools.permutations(BELL):
                    best = max(best, local_search(Mf[i], Mf[i], [BELL[k] for k in pm], obs_set, obs_set, restarts=6)[0])
            res[tol] = (best, len(obs_set))
        print(f'B = qF = {B}: делителей {len(Ts)}; потери: медиана {np.median(loss):.1e}, доля < 1e−3: {np.mean(loss < 1e-3):.2f}; '
              f'«комплексных» событий до {mixed:.0%}; |y| наблюдаемых макс {ycomp.max():.3f}; C макс {conc.max():.3f}; '
              f'Renou без пластинок: T = {res[1e-3][0] if res[1e-3][0] is None else round(res[1e-3][0], 3)} (потери < 1e−3, {res[1e-3][1]} настроек), '
              f'{res[1e-2][0] if res[1e-2][0] is None else round(res[1e-2][0], 3)} (потери < 1e−2, {res[1e-2][1]})')
        sys.stdout.flush()
