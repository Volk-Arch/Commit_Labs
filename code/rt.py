#!/usr/bin/env python3
"""Рю–Такаянаги в каузальном множестве: сильная субаддитивность.

ПОВОД. holo.py: восстановимости нет — граничный экран не лучше
случайного той же величины (0.585 против 0.589 при N=2000, 0.682 против
0.676 при N=4000). Значит площадной закон и граничность остатка — это
совпадения по СЧЁТУ, а не голография.

Но у голографии есть ещё одна проверяемая подпись, и она сильнее
площадного закона. Содержание формулы Рю–Такаянаги не в том, что энтропия
равна площади, а в том, что площадь берётся МИНИМАЛЬНАЯ по поверхностям,
натянутым на край области. У минимума есть следствие, которое площадной
закон сам по себе не даёт:

    СИЛЬНАЯ СУБАДДИТИВНОСТЬ    S(AuB) + S(AnB) <= S(A) + S(B)

RT выполняет её всегда (минимальные поверхности можно переразрезать и
склеить), а объёмный закон её нарушает. Поэтому это настоящая проверка.

ЧТО СЧИТАЕМ. Энтропия области по Доу–Соркину: число СВЯЗЕЙ (пар без
ничего между), пересекающих границу области причинной зависимости D(A).

ОБЛАСТИ — ПЛАСТЫ по x1 на срезе t0, потому что у пластов объединение и
пересечение тоже пласты, и SSA проверяется без возни с формой:

    A = [a1,b1], B = [a2,b2],  a1<a2<b1<b2
    AuB = [a1,b2],  AnB = [a2,b1]

Область причинной зависимости пласта [a,b] при t0:

    D = { (t,x) : a + |t-t0| <= x1 <= b - |t-t0| }

Проверяется также обычная субаддитивность S(AuB) <= S(A) + S(B) и
масштабирование: у пласта граница — две плоскости постоянной площади,
значит S не должна зависеть от ширины, когда та много больше масштаба
дискретности.

  python3 rt.py
  python3 rt.py --n 6000 --seeds 5
"""
import argparse
import time

import numpy as np

T = 2.0
T0 = 1.0


def diamond(N, rs):
    out, got = [], 0
    while got < N:
        m = int((N - got) * 9) + 64
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        ok = (Y[:, 0] >= r) & (T - Y[:, 0] >= r)
        if ok.any():
            out.append(Y[ok])
            got += int(ok.sum())
    return np.vstack(out)[:N]


def cone_rel(X, chunk=400):
    N = len(X)
    R = np.empty((N, N), bool)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        dt = X[a:b, None, 0] - X[None, :, 0]
        dx = np.linalg.norm(X[a:b, None, 1:] - X[None, :, 1:], axis=-1)
        R[a:b] = dt > dx
    np.fill_diagonal(R, False)
    return R


def slab(X, a, b):
    """Область причинной зависимости пласта [a,b] по x1 на срезе t0."""
    w = np.abs(X[:, 0] - T0)
    return (X[:, 1] >= a + w) & (X[:, 1] <= b - w)


def cross_links(R, masks, chunk=400):
    """Число связей, пересекающих границу каждой области, за один проход."""
    N = len(R)
    A = R.astype(np.float32)
    cnt = np.zeros(len(masks))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        btw = A[s:e] @ A
        L = R[s:e] & (btw == 0)
        for k, m in enumerate(masks):
            # связь (i,j): ровно одно из i,j внутри области
            cnt[k] += float((L & (m[s:e, None] ^ m[None, :])).sum())
    return cnt


def main():
    ap = argparse.ArgumentParser(description='SSA в каузальном множестве')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--a1', type=float, default=-0.50)
    ap.add_argument('--a2', type=float, default=-0.12)
    ap.add_argument('--b1', type=float, default=0.12)
    ap.add_argument('--b2', type=float, default=0.50)
    a = ap.parse_args()

    t0 = time.time()
    rho = a.n / (np.pi * T ** 4 / 24)
    ell = (24 / (rho * np.pi)) ** 0.25
    print(f'ромб T={T}, N={a.n}, плотность {rho:.0f}')
    print(f'масштаб связи ~{ell:.3f}; пласты шириной '
          f'{a.b1-a.a1:.2f} и {a.b2-a.a2:.2f}, '
          f'объединение {a.b2-a.a1:.2f}, пересечение {a.b1-a.a2:.2f}')
    print()

    rows = []
    for t in range(a.seeds):
        rs = np.random.default_rng(3300 + 17 * t)
        X = diamond(a.n, rs)
        R = cone_rel(X)
        masks = [slab(X, a.a1, a.b1), slab(X, a.a2, a.b2),
                 slab(X, a.a1, a.b2), slab(X, a.a2, a.b1)]
        rows.append(cross_links(R, masks))
        del R, X
    V = np.array(rows)
    nm = ['S(A)', 'S(B)', 'S(AuB)', 'S(AnB)']
    print('ЭНТРОПИИ (связей через границу области причинной зависимости)')
    for i, n in enumerate(nm):
        print(f'  {n:>8}{V[:,i].mean():>12.1f} ±{V[:,i].std()/np.sqrt(len(V)):<7.1f}')

    ssa = V[:, 0] + V[:, 1] - V[:, 2] - V[:, 3]
    sa = V[:, 0] + V[:, 1] - V[:, 2]
    print()
    print('СИЛЬНАЯ СУБАДДИТИВНОСТЬ: S(A)+S(B) - S(AuB) - S(AnB) >= 0 ?')
    m, e = ssa.mean(), ssa.std() / np.sqrt(len(ssa))
    print(f'  {m:>12.1f} ±{e:<7.1f}   ({m/max(e,1e-9):>5.1f} сигма, '
          f'{"ВЫПОЛНЕНА" if m > 0 else "НАРУШЕНА"})')
    print(f'  в долях от S(A)+S(B): {m/(V[:,0]+V[:,1]).mean():.4f}')
    print()
    print('СУБАДДИТИВНОСТЬ: S(A)+S(B) - S(AuB) >= 0 ?')
    m2, e2 = sa.mean(), sa.std() / np.sqrt(len(sa))
    print(f'  {m2:>12.1f} ±{e2:<7.1f}   '
          f'({"ВЫПОЛНЕНА" if m2 > 0 else "НАРУШЕНА"})')
    print()
    print('ЗАВИСИМОСТЬ ОТ ШИРИНЫ (у пласта граница - две плоскости, '
          'площадь от ширины не зависит)')
    print(f"{'ширина':>8}{'S':>14}{'ошибка':>9}")
    for w in (0.2, 0.3, 0.4, 0.6, 0.8):
        vs = []
        for t in range(a.seeds):
            rs = np.random.default_rng(3300 + 17 * t)
            X = diamond(a.n, rs)
            R = cone_rel(X)
            vs.append(cross_links(R, [slab(X, -w / 2, w / 2)])[0])
            del R, X
        vs = np.array(vs)
        print(f'{w:>8.1f}{vs.mean():>14.1f}{vs.std()/np.sqrt(len(vs)):>9.1f}')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
