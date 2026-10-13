#!/usr/bin/env python3
"""ДВУХТОЧЕЧНЫЙ ИНТЕГРАЛ: <N_m>(N) без графа и без матрицы отношений.

Задача. Условие пересмотра закрытого маршрута (`sigma-doc.md`) требует
знать, где показатель числа вкладывающих пар `P` подходит к 1.5.
Замер при `N ~ 10^5` матрицей невозможен (10^10 элементов). Но среднее
число пар с ровно `m` промежуточными элементами есть ТОЧНОЕ выражение

    <N_m> = rho^2 ∫∫_{x<y} e^{-lam} lam^m/m! d4x d4y,    lam = rho V(x,y)

а в алмазе с T=1 объём интервала равен V_T tau^4, значит lam = N tau^4
РОВНО. То есть N входит одним числом, и один интеграл даёт все N сразу:
от 10^2 до 10^8, без сприклинга вообще.

Оценщик по M парам из U(D) x U(D), ориентированным по времени:

    <N_m> = (N^2/2) * (1/M) sum_{связанные} e^{-lam} lam^m/m!

Множитель 1/2 потому, что ориентирование считает и x<y, и y<x.

АНАЛИТИЧЕСКАЯ АСИМПТОТИКА (встроенный контроль). Мера пар по
собственному времени ведёт себя как q(tau) -> a*tau при малых tau
(объём пришахтного слоя растёт как tau^2). Подстановка u = N^{1/4} tau
даёт

    <N_m> -> (a/8) * Gamma(m+1/2)/m! * N^{3/2}

то есть показатель РОВНО 3/2, а смесь весов стремится к отношениям
Gamma(m+1/2)/m! = 1 : 1/2 : 3/8 : 5/16.

И главное: веса Бенинказы–Доукера обязаны гасить ведущий член,

    sum_m w_m Gamma(m+1/2)/m! = sqrt(pi) * (-1 + 9/2 - 6 + 5/2) = 0

ТОЧНО. Это проверяется символически и объясняет, почему <S> в замерах
болтается около нуля без роста (50, 102, 192, 249, 156).

ТОЧНОСТЬ. tau^2 = dt^2 - dr^2 при малых tau есть разность близких
чисел: при tau ~ 10^-2 и dt ~ 0.5 во float32 остаётся три знака.
Поэтому всё в float64 и в устойчивой форме (dt-dr)(dt+dr); расхождение
с float32 замеряется отдельным контролем.
"""
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)
WT = np.array([-1.0, 9.0, -16.0, 8.0])
MFAC = np.array([1.0, 1.0, 2.0, 6.0])


def uniform_diamond(m, rs, dtype=np.float64):
    """Равномерно в алмазе {|x| < min(t, 1-t)}, отбор 25%."""
    out, got = [], 0
    while got < m:
        k = int((m - got) * 5) + 4096
        t = rs.random(k).astype(dtype)
        v = rs.normal(size=(k, 3)).astype(dtype)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(k).astype(dtype) ** (1.0 / 3.0)) * dtype(0.5)
        ok = rad < np.minimum(t, 1 - t)
        if ok.any():
            out.append(np.column_stack([t[ok], (v * rad[:, None])[ok]]))
            got += int(ok.sum())
    return np.vstack(out)[:m]


def taus(npair, rs, dtype=np.float64):
    """tau для СВЯЗАННЫХ пар из npair независимых пар; устойчивая форма."""
    A = uniform_diamond(npair, rs, dtype)
    B = uniform_diamond(npair, rs, dtype)
    dt = np.abs(B[:, 0] - A[:, 0])
    dr = np.linalg.norm(B[:, 1:] - A[:, 1:], axis=1)
    t2 = (dt - dr) * (dt + dr)
    ok = t2 > 0
    return np.sqrt(t2[ok]), npair


def nm_direct(tau, M, Ns):
    """<N_m>(N) прямым несмещённым оценщиком по выборке tau."""
    out = np.zeros((len(Ns), 4))
    for i, N in enumerate(Ns):
        lam = N * tau ** 4
        e = np.exp(-lam)
        for m in range(4):
            out[i, m] = (N * N / 2.0) * (e * lam ** m / MFAC[m]).sum() / M
    return out


def main():
    print(__doc__)
    rs = np.random.default_rng(1618033)

    # --- сбор меры q(tau) ---
    CH, NCH = 2_000_000, 20
    tau_all, Mtot = [], 0
    for _ in range(NCH):
        tt, mm = taus(CH, rs)
        tau_all.append(tt.astype(np.float64))
        Mtot += mm
    tau = np.concatenate(tau_all)
    print(f'собрано: {Mtot:,} пар, из них связанных {len(tau):,}')
    r_meas = 2.0 * len(tau) / Mtot
    print(f'\nКОНТРОЛЬ A: доля связанных (Майрхейм–Майер для d=4)')
    print(f'  замерено {r_meas:.5f} ± {2*math.sqrt(len(tau))/Mtot:.5f}'
          f'  против точного 0.1  '
          f'({abs(r_meas-0.1)/(2*math.sqrt(len(tau))/Mtot):.1f} сигм)')

    print('\nКОНТРОЛЬ B: тождество весов (символически)')
    g = np.array([math.gamma(m + 0.5) for m in range(4)]) / MFAC
    print('  Gamma(m+1/2)/m! = ' + ', '.join(f'{v:.6f}' for v in g))
    print('  отношения       = ' + ', '.join(f'{v/g[0]:.6f}' for v in g)
          + '   (точно 1 : 1/2 : 3/8 : 5/16)')
    print(f'  sum w_m Gamma(m+1/2)/m! = {float(WT @ g):.3e}  '
          f'(обязано быть 0 — это и есть устройство действия БД)')

    print('\nКОНТРОЛЬ C: float32 против float64 на малых tau')
    t32, _ = taus(400_000, rs, np.float32)
    for lo, hi in ((0.3, 0.5), (0.03, 0.05), (0.01, 0.02)):
        f64 = ((tau > lo) & (tau < hi)).sum() / len(tau)
        f32 = ((t32 > lo) & (t32 < hi)).sum() / len(t32)
        print(f'  доля tau в [{lo}, {hi}]: f64 {f64:.3e}, '
              f'f32 {f32:.3e}, расхождение '
              f'{abs(f32-f64)/max(f64,1e-12)*100:5.1f}%')

    # --- поведение меры при малых tau: q(tau) -> a tau ---
    print('\nМЕРА ПО tau: проверка q(tau)/tau -> a')
    edges = np.logspace(-3.0, 0.0, 61)
    cnt, _ = np.histogram(tau, bins=edges)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    wid = np.diff(edges)
    q = cnt / Mtot / wid
    sel = (cnt > 400) & (ctr < 0.3)
    print(f"{'tau':>9}{'q(tau)':>12}{'q/tau':>10}{'±':>8}")
    for i in np.where(sel)[0][::3]:
        print(f'{ctr[i]:>9.4f}{q[i]:>12.4f}{q[i]/ctr[i]:>10.4f}'
              f'{q[i]/ctr[i]/math.sqrt(cnt[i]):>8.4f}')
    wq = q[sel] / ctr[sel]
    we = wq / np.sqrt(cnt[sel])
    a = float((wq / we ** 2).sum() / (1 / we ** 2).sum())
    ae = float(1 / math.sqrt((1 / we ** 2).sum()))
    sl = np.polyfit(np.log(ctr[sel]), np.log(wq), 1)[0]
    print(f'  a = {a:.4f} ± {ae:.4f}, наклон q/tau по tau = {sl:+.4f} '
          f'(плоско => закон q ~ a tau подтверждён)')

    # --- асимптотические постоянные ---
    A_m = (a / 8.0) * g
    print(f'\nАСИМПТОТИКА: <N_m> -> A_m N^1.5, '
          f'A_m = ' + ', '.join(f'{v:.4f}' for v in A_m))
    print(f'  сумма A_m = {A_m.sum():.4f}  => P -> {A_m.sum():.4f} N^1.5')
    print(f'  sum w_m^2 A_m = {float((WT**2) @ A_m):.3f}  => '
          f'sigma_шум -> k4 sqrt(...) = '
          f'{K4*math.sqrt(float((WT**2) @ A_m)):.3f} N^0.75')

    # --- прямой оценщик на замеренных N ---
    print('\nКОНТРОЛЬ D: интеграл против прямого замера сприклингом')
    arch = {200: (1188, 302, 153, 93), 400: (3827, 1142, 634, 413),
            800: (12411, 4195, 2478, 1719),
            1600: (38433, 14109, 8790, 6346),
            3200: (118961, 47169, 30669, 22959)}
    Ns = sorted(arch)
    got = nm_direct(tau, Mtot, Ns)
    print(f"{'N':>6}" + ''.join(f'{f"N{m}":>19}' for m in range(4)))
    for i, N in enumerate(Ns):
        cells = []
        for m in range(4):
            cells.append(f'{got[i,m]:8.0f}/{arch[N][m]:<6d}'
                         f'{got[i,m]/arch[N][m]:5.2f}')
        print(f'{N:>6}' + ''.join(f'{c:>19}' for c in cells))
    np.save('/tmp/claude-0/tau_sample.npy', tau[::7])
    print(f'  (интеграл/замер; отклонения — конечность одного '
          f'сприклинга и его границы)')

    # --- то, за чем всё делалось ---
    print('\n=== ПОКАЗАТЕЛЬ P: где он подходит к 1.5 ===')
    grid = np.array([10 ** (2 + 0.25 * i) for i in range(25)])
    nmv = nm_direct(tau, Mtot, grid)
    P = nmv.sum(axis=1)
    print(f"{'N':>12}{'P':>14}{'P/N^1.5':>10}{'местный пок.':>14}"
          f"{'смесь N1/N0':>12}{'N3/N0':>8}")
    prev = None
    for i, N in enumerate(grid):
        loc = (math.log(P[i] / P[i - 1]) / math.log(grid[i] / grid[i - 1])
               if i else float('nan'))
        if i % 2 == 0 or i == len(grid) - 1:
            print(f'{N:>12.3g}{P[i]:>14.4g}{P[i]/N**1.5:>10.4f}'
                  f'{loc:>14.4f}{nmv[i,1]/nmv[i,0]:>12.4f}'
                  f'{nmv[i,3]/nmv[i,0]:>8.4f}')
        prev = loc
    for tgt in (1.60, 1.55, 1.52, 1.51):
        locs = [(math.log(P[i] / P[i - 1])
                 / math.log(grid[i] / grid[i - 1]), grid[i])
                for i in range(1, len(grid))]
        hit = next((n for v, n in locs if v <= tgt), None)
        print(f'  показатель P <= {tgt}: при N ~ '
              f'{hit:.3g}' if hit else
              f'  показатель P <= {tgt}: не достигнут до N = 10^8')


if __name__ == '__main__':
    main()
