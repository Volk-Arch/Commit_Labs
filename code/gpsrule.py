#!/usr/bin/env python3
"""Квадратичное правило против монотонного на ОДНИХ И ТЕХ ЖЕ пяти числах.

Пять «спутников» — часы на неподвижных мировых линиях, тикают с шагом Δ.
Каждому событию x из порядка достаётся пять чисел:
    p_i(x) = номер последнего тика спутника i в ПРОШЛОМ x
(только прошлое — годится для роста; это «GPS observables» Ровелли,
то есть координаты излучения). По причинному условию тик k раньше x
тогда и только тогда, когда t_x − t_k > |x − a_i|; отсюда p_i = floor(...).

Из пяти чисел два правила порядка:
  ЯЩИК (монотонное):  b ≺ x, если τ_i(b) < τ_i(x) для всех i
                       (берём ТОЧНЫЕ τ — самый выгодный для ящика случай)
  GPS  (квадратичное): восстановить точку из пяти τ и сравнить знак
                       интервала  Δt > |Δx|

Сравнение с ИСТИННЫМ порядком на тех же точках.
"""
import math, sys, time
import numpy as np
sys.path.insert(0, '/home/claude')
from core import frac_pairs, dimension, height, links_of, spectral, rel_of

V4 = 2 * math.pi / 3        # 4-объём ромба |t|+|x|<=1

def sprinkle(N, rs):
    out = []
    while sum(len(o) for o in out) < N:
        c = rs.uniform(-1, 1, (4 * N, 4))
        ok = np.abs(c[:, 0]) + np.linalg.norm(c[:, 1:], axis=1) <= 1
        out.append(c[ok])
    return np.concatenate(out)[:N]

def satellites(rs):
    # тетраэдр радиуса 1.6 + пятый на оси, всё неподвижно; повёрнуто случайно
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / math.sqrt(3)
    a = np.vstack([1.6 * v, [[0, 0, 2.2]]])
    Q, _ = np.linalg.qr(rs.normal(size=(3, 3)))
    return a @ Q

def minkowski(T, Xs):
    dt = T[:, None] - T[None, :]
    dx = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=-1)
    R = dt > dx
    np.fill_diagonal(R, False)
    return R

def reconstruct(tau, a):
    """Точка из пяти времён излучения: линейное начальное + Гаусс–Ньютон."""
    n = len(tau)
    A = np.zeros((n, 4, 4)); b = np.zeros((n, 4))
    for r, i in enumerate(range(1, 5)):
        A[:, r, 0] = -2 * (tau[:, i] - tau[:, 0])
        A[:, r, 1:] = 2 * (a[i] - a[0])
        b[:, r] = (a[i] @ a[i] - a[0] @ a[0]) - tau[:, i] ** 2 + tau[:, 0] ** 2
    Y = np.linalg.lstsq if False else None
    Y = np.linalg.solve(A + 1e-12 * np.eye(4), b[..., None])[..., 0]   # (t, x, y, z)
    bad = ~np.isfinite(Y).all(1) | (np.abs(Y) > 3).any(1)
    Y[bad] = 0.0                                           # плохой старт -> центр
    for _ in range(8):
        d = Y[:, None, 1:] - a[None]                       # n,5,3
        rr = np.linalg.norm(d, axis=-1)
        res = Y[:, :1] - tau - rr                          # n,5
        J = np.concatenate([np.ones((n, 5, 1)), -d / rr[..., None]], -1)
        JT = np.swapaxes(J, 1, 2)
        H = JT @ J + 1e-9 * np.eye(4)
        step = np.linalg.solve(H, (JT @ res[..., None]))[..., 0]
        step = np.where(np.isfinite(step), step, 0.0)
        Y = Y - np.clip(step, -0.5, 0.5)
    return Y

def compare(R, R0, N):
    R0n = R0.sum()
    return (R & ~R0).sum() / R0n, (R0 & ~R).sum() / R0n, ((R ^ R0).sum()) / N

def stats(R, spec):
    L = links_of(R)
    return (frac_pairs(R), dimension(R=R), height(R)[0], L.sum() / len(R),
            spectral(R) if spec else float('nan'))

def one(N, rel_delta, seed, eph=0.0, spec=True):
    rs = np.random.default_rng(seed)
    X = sprinkle(N, rs); a = satellites(rs)
    ell = (V4 / N) ** 0.25; D = rel_delta * ell
    t0 = -6 + D * rs.uniform(size=5)                       # фазы часов
    tau_exact = X[:, :1] - np.linalg.norm(X[:, None, 1:] - a[None], axis=-1)
    p = np.floor((tau_exact - t0) / D)                     # из порядка: последний тик в прошлом
    tau_hat = t0 + (p + 0.5) * D
    a_cal = a + rs.normal(scale=eph * D, size=a.shape)     # погрешность эфемерид ~ Δ
    t_cal = rs.normal(scale=eph * D, size=5)
    Y = reconstruct(tau_hat - t_cal, a_cal)
    R0 = minkowski(X[:, 0], X[:, 1:])
    Rg = minkowski(Y[:, 0], Y[:, 1:])
    Rb = rel_of(tau_exact)
    pos_err = np.median(np.linalg.norm(Y - X, axis=1)) / ell
    return R0, Rg, Rb, pos_err

if __name__ == '__main__':
    t00 = time.time()
    print('ПРОВЕРКА: точные τ -> GPS должен совпасть с истиной')
    rs = np.random.default_rng(1); X = sprinkle(800, rs); a = satellites(rs)
    te = X[:, :1] - np.linalg.norm(X[:, None, 1:] - a[None], axis=-1)
    Y = reconstruct(te, a)
    print(f'  макс. ошибка положения {np.abs(Y - X).max():.2e}, '
          f'различий в порядке {(minkowski(Y[:,0],Y[:,1:]) ^ minkowski(X[:,0],X[:,1:])).sum()}')
    R0 = minkowski(X[:, 0], X[:, 1:])
    print(f'  ящик на точных τ нарушает «истина ⊂ ящик»: {(R0 & ~rel_of(te)).sum()} (ждём 0)\n')

    hdr = f"{'':>22}{'лишних/R':>10}{'пропущ/R':>10}{'ошибок/соб':>11}{'доля':>8}{'разм':>7}{'выс':>6}{'связ/соб':>9}{'спектр':>9}"
    for N in (1000, 2000, 4000):
        spec = N <= 2000
        print(f'N = {N}   (ℓ = {(V4/N)**0.25:.3f})'); print(hdr)
        rows = {}
        for lab, rd, eph in (('GPS Δ=ℓ', 1.0, 0.0), ('GPS Δ=ℓ, эфемериды±', 1.0, 0.5),
                             ('GPS Δ=0.3ℓ', 0.3, 0.0), ('GPS Δ=0.1ℓ', 0.1, 0.0)):
            acc = []
            for s in range(3):
                R0, Rg, Rb, pe = one(N, rd, 10 * s + N, eph, spec)
                acc.append(compare(Rg, R0, N) + stats(Rg, spec) + (pe,))
                if lab == 'GPS Δ=ℓ':
                    rows.setdefault('ИСТИНА', []).append((0, 0, 0) + stats(R0, spec) + (0,))
                    rows.setdefault('ЯЩИК (точные τ)', []).append(compare(Rb, R0, N) + stats(Rb, spec) + (0,))
            rows[lab] = acc
        for lab in ('ИСТИНА', 'ЯЩИК (точные τ)', 'GPS Δ=ℓ', 'GPS Δ=ℓ, эфемериды±', 'GPS Δ=0.3ℓ', 'GPS Δ=0.1ℓ'):
            m = np.nanmean(np.array(rows[lab], float), 0)
            print(f'{lab:>22}{m[0]:>10.4f}{m[1]:>10.4f}{m[2]:>11.2f}{m[3]:>8.4f}{m[4]:>7.2f}'
                  f'{m[5]:>6.1f}{m[6]:>9.2f}{m[7]:>+9.4f}' + (f'   полож.ош. {m[8]:.3f}ℓ' if m[8] else ''))
        print(f'   [{time.time()-t00:.0f} с]\n'); sys.stdout.flush()
