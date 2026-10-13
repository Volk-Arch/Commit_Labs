#!/usr/bin/env python3
"""ПОРЯДКОВАЯ ПЛОЩАДЬ: разведение вкладов по k, с выводом, а не догадкой.

ПОЧЕМУ `F_0` НЕ ПЛОЩАДЬ (поймано за 19 секунд в `area.py`). Показатель
вышел 0.586 +- 0.010 вместо 1/2, и отношение к `pi tau²` плыло по
полосам (0.457 -> 0.544). Диагноз: площадь талии идёт как `tau²`, то
есть `C_1^(1/2)`, а трёхмерный объём светового конуса как `tau³`, то
есть `C_1^(3/4)`; 0.586 лежит между, и значит `F_0` есть СМЕСЬ.

ВЫВОД КОНСТРУКЦИИ. Элемент имеет ровно `k` событий в будущем с
пуассоновской вероятностью, поэтому

    <F_k> = ∫ (u^k e^(-u)/k!) g(u/rho) du,    g(V_f) = dV/dV_f

где `V_f(x)` — объём от `x` до остриё `f`. Вблизи конуса `V_f -> 0`, и
нужна степень `g`. Считаем: `V_f = (pi/24) tau⁴`, а четырёхмерный объём
слоя `{tau(x,f) <= s}` есть `∫_0^s tau³ H(tau) dtau`, где `H` —
гиперболический объём части гиперболоида внутри будущего `p`. При
`tau -> 0` гиперболоид прижимается к конусу и `H` РАСХОДИТСЯ как
`(T/tau)²` (гиперболический объём растёт как `e^(2eta)`, а предельная
быстрота идёт как `log(T/tau)`). Отсюда

    V(tau <= s)  ~  T² s²        то есть ПЛОЩАДЬ на s²

и `g(w) ~ C w^(-1/2)` с амплитудой `C ∝ T²`. Подставляя,

    <F_k>  =  a * Г(k+1/2)/k!  +  поправки,     a ∝ (T/l)²

Поправки идут от следующих членов разложения `g`, и все они суть
полуцелые сдвиги:

    phi_j(k) = Г(k + 1/2 + j/2) / Г(k+1),    j = 0, 1, 2, 3
    phi_0 ~ k^(-1/2)   phi_1 = 1   phi_2 ~ k^(1/2)   phi_3 = k+1

Значит ЗАВИСИМОСТЬ ОТ k РАЗВОДИТ вклады, и амплитуда при `phi_0` есть
площадь. Формула из статьи Бака, Доукера, Джабба и Сурьи при этом не
нужна — конструкция выведена здесь.

ПРОВЕРКИ, в порядке строгости:
  1. КОНТРОЛЬ ВЫВОДА в непрерывном: доля точек плоского ромба с
     `tau(x,f) <= s` обязана расти как `s²`, а не `s⁴`. Это основание
     всей конструкции, и оно проверяется без всякого сприклинга.
  2. амплитуда `a` обязана идти как `C_1^(1/2)` РОВНО — это площадь;
  3. отношение `a` к площади талии `pi tau²/4` обязано быть ПОСТОЯННЫМ
     по полосам размера (у `F_0` оно плыло на 19 %);
  4. ошибка — складным ножом по ЗАСЕВАМ.

  python3 joint.py [--seeds N] [--per M]
"""
import argparse
import math
import time

import numpy as np

from chains import fits
from pencil import sprinkle

KMAX = 12


def degrees(t, S, block=256):
    """Числа событий в прошлом и будущем каждого элемента интервала.

    ПОЧЕМУ БЛОКАМИ И ЧЕРЕЗ ГРАМА. Прямое `S[None,:,:] - S[:,None,:]`
    требует массива `C_1 x C_1 x 3`, то есть 138 МБ при `C_1 = 2400`, и
    первый прогон был убит по памяти. Квадрат расстояния берётся из
    матрицы Грама,

        |x_i - x_j|² = s_i + s_j - 2 (X Xᵀ)_ij,

    что убирает множитель 3, а блоки по строкам ограничивают пик.
    """
    n = len(t)
    sq = np.einsum('ij,ij->i', S, S)
    pas = np.zeros(n, dtype=np.int64)
    fut = np.zeros(n, dtype=np.int64)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d2 = sq[i0:i1, None] + sq[None, :] - 2.0 * (S[i0:i1] @ S.T)
        dt = t[None, :] - t[i0:i1, None]
        R = (dt > 0) & (dt * dt > d2)
        fut[i0:i1] = R.sum(1)
        pas += R.sum(0)
        del d2, dt, R
    return pas, fut


def phi(j, k):
    """Г(k + 1/2 + j/2)/Г(k+1), устойчиво через логарифмы."""
    return np.exp(np.array([math.lgamma(kk + 0.5 + j / 2.0)
                            - math.lgamma(kk + 1.0) for kk in k]))


def control_continuum(n=3000000, rs=None):
    """КОНТРОЛЬ №1: доля точек ромба с tau(x,f) <= s растёт как s^m.

    Если вывод верен, m = 2 (площадь), а не 4 (как было бы, если бы
    слой имел постоянную толщину по собственному времени).

    ПОЙМАННАЯ ОШИБКА: в первой версии отбор шёл циклом `while len(got) <
    n`, где `got` — СПИСОК массивов, так что `len` считал массивы, а не
    точки. Цикл выделял по сто пятьдесят тысяч массивов и убивал процесс
    по памяти. Здесь один проход без цикла.
    """
    rs = rs or np.random.default_rng(12345)
    T = 1.0
    t = rs.uniform(0, T, n)
    x = rs.uniform(-T / 2, T / 2, (n, 3))
    r2 = np.einsum('ij,ij->i', x, x)
    ok = (t ** 2 > r2) & ((T - t) ** 2 > r2)
    t, r2 = t[ok], r2[ok]
    tau = np.sqrt(np.maximum((T - t) ** 2 - r2, 0.0))
    ss = np.array([0.02, 0.03, 0.05, 0.08, 0.12, 0.18])
    frac = np.array([(tau <= s).mean() for s in ss])
    p = np.polyfit(np.log(ss), np.log(np.maximum(frac, 1e-12)), 1)[0]
    return ss, frac, p, int(ok.sum())


def harvest(cfg, rs, want):
    L, lo_c, hi_c, eta0, band = cfg
    X = sprinkle('flat', 0.0, L, 2 * eta0, eta0, rs)
    t = np.ascontiguousarray(X[:, 0])
    S = np.ascontiguousarray(X[:, 1:])
    tlo = (24.0 * lo_c / math.pi) ** 0.25
    thi = (24.0 * hi_c / math.pi) ** 0.25
    out = []
    guard = 0
    while len(out) < want and guard < want * 30:
        guard += 1
        ip = int(rs.integers(np.searchsorted(t, eta0 - band - thi * 1.4),
                             np.searchsorted(t, eta0 + band)))
        p0, ps = t[ip], S[ip]
        j0 = np.searchsorted(t, p0 + tlo)
        j1 = np.searchsorted(t, p0 + thi * 2.2)
        if j1 - j0 < 40:
            continue
        de = t[j0:j1] - p0
        d = S[j0:j1] - ps
        dx2 = np.einsum('ij,ij->i', d, d)
        tau2 = de * de - dx2
        ok = np.flatnonzero((tau2 > tlo * tlo) & (tau2 < thi * thi))
        if len(ok) == 0:
            continue
        for q in rs.permutation(ok)[:6]:
            if len(out) >= want:
                break
            jj = int(q)
            f0, fs = t[j0 + jj], S[j0 + jj]
            if abs(0.5 * (p0 + f0) - eta0) > band:
                continue
            if not fits(np.concatenate([[p0], ps]),
                        np.concatenate([[f0], fs]), L, 2 * eta0):
                continue
            k0 = np.searchsorted(t, p0, 'right')
            k1 = np.searchsorted(t, f0, 'left')
            if k1 - k0 < lo_c:
                continue
            Sl, tl = S[k0:k1], t[k0:k1]
            u = Sl - ps
            w = fs - Sl
            m = (((tl - p0) ** 2 > np.einsum('ij,ij->i', u, u))
                 & ((f0 - tl) ** 2 > np.einsum('ij,ij->i', w, w)))
            a = int(m.sum())
            if not (lo_c <= a <= hi_c):
                continue
            pas, fut = degrees(tl[m], Sl[m])
            hf = np.bincount(fut, minlength=KMAX + 1)[:KMAX + 1]
            hp = np.bincount(pas, minlength=KMAX + 1)[:KMAX + 1]
            out.append([a, math.sqrt(max(tau2[jj], 1e-12))]
                       + list(hf) + list(hp))
    del X, t, S
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=14)
    ap.add_argument('--per', type=int, default=450)
    a_ = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 74)
    print('КОНТРОЛЬ №1 (непрерывный, без сприклинга): V(tau<=s) ~ s^m')
    ss, frac, p, nin = control_continuum()
    print(f'  точек внутри ромба: {nin}')
    print(f"{'s/T':>8}{'доля объёма':>14}{'доля/s²':>12}")
    for s, fr in zip(ss, frac):
        print(f'{s:>8.3f}{fr:>14.6f}{fr/s**2:>12.4f}')
    print(f'  показатель m = {p:.4f}   (вывод предсказывает 2, '
          f'наивное «слой постоянной толщины» дало бы 4)')
    print(f'  => {"ВЫВОД ПОДТВЕРЖДЁН" if abs(p-2) < 0.08 else "ВЫВОД НЕ ПОДТВЕРЖДЁН"}')
    print()

    cfg = (26.0, 120, 1600, 10.0, 4.0)
    rows, sdi = [], []
    for s in range(a_.seeds):
        rs = np.random.default_rng(773000 + 8819 * s)
        r = harvest(cfg, rs, a_.per)
        rows += r
        sdi += [s] * len(r)
    A = np.array(rows, dtype=float)
    sd = np.array(sdi)
    print(f'плоское: {len(A)} интервалов, C_1 от {A[:,0].min():.0f} до '
          f'{A[:,0].max():.0f}   ({time.time()-t0:.0f} с)')
    print()
    ks = np.arange(KMAX + 1)
    q = np.quantile(A[:, 0], np.linspace(0, 1, 6))
    NB = [4]

    def fit_band(mask, nb=None):
        """Амплитуды c_j по профилю <F_k> (будущее и прошлое вместе).

        Длина базиса — параметр: подведущие члены разложения `g` суть
        полуцелые сдвиги, и если базис короток, часть их протекает в
        `c_0`. Сходимость показателя к 1/2 при росту базиса и есть
        проверка того, что это именно подведущие члены, а не иное.
        """
        nb = nb or NB[0]
        B = np.column_stack([phi(j, ks) for j in range(nb)])
        y = np.concatenate([A[mask][:, 2:2 + KMAX + 1].mean(0),
                            A[mask][:, 3 + KMAX:].mean(0)])
        D = np.vstack([B, B])
        w = 1.0 / np.maximum(y, 1.0)          # пуассоновская дисперсия
        Dw = D * w[:, None]
        c = np.linalg.solve(D.T @ Dw, Dw.T @ y)
        r = y - D @ c
        return c, float((w * r * r).sum()), len(y) - nb

    print('АМПЛИТУДЫ ПО ПОЛОСАМ РАЗМЕРА. c_0 — при phi_0 ~ k^(-1/2),')
    print('и это площадь. Сверяем с площадью талии pi tau²/4.')
    print(f"{'C_1':>8}{'шт':>7}{'c_0':>10}{'ст.ош':>9}{'pi tau²/4':>12}"
          f"{'c_0/площадь':>14}{'c_1':>10}{'c_2':>10}")
    bands = []
    for i in range(5):
        m = (A[:, 0] >= q[i]) & (A[:, 0] <= q[i + 1])
        if m.sum() < 40:
            continue
        c = fit_band(m)[0]
        us = np.unique(sd)
        jk = np.array([fit_band(m & (sd != u))[0] for u in us])
        kk = len(us)
        e0 = math.sqrt((kk - 1) / kk * ((jk[:, 0] - jk[:, 0].mean()) ** 2).sum())
        ar = math.pi * (A[m][:, 1] ** 2).mean() / 4.0
        bands.append((A[m][:, 0].mean(), c[0], e0, ar))
        print(f'{A[m][:,0].mean():>8.0f}{m.sum():>7}{c[0]:>10.2f}{e0:>9.2f}'
              f'{ar:>12.2f}{c[0]/ar:>14.4f}{c[1]:>10.2f}{c[2]:>10.2f}')
    print()
    Bd = np.array(bands)
    # показатель со складным ножом по засевам: лестница по полосам
    def expo(keep):
        pts = []
        for i in range(5):
            m = keep & (A[:, 0] >= q[i]) & (A[:, 0] <= q[i + 1])
            if m.sum() < 40:
                continue
            pts.append((A[m][:, 0].mean(), fit_band(m)[0][0]))
        P = np.array(pts)
        return np.polyfit(np.log(P[:, 0]), np.log(P[:, 1]), 1)[0]
    us = np.unique(sd)
    pw = expo(np.ones(len(A), bool))
    jkp = np.array([expo(sd != u) for u in us])
    kk = len(us)
    epw = math.sqrt((kk - 1) / kk * ((jkp - jkp.mean()) ** 2).sum())
    rat = Bd[:, 1] / Bd[:, 3]
    print(f'  ПОКАЗАТЕЛЬ c_0 по C_1: {pw:.4f} +- {epw:.4f}'
          f'   (площадь требует 0.5000,'
          f' отклонение {abs(pw-0.5)/max(epw,1e-9):.1f} сигм)')
    print(f'  отношение c_0/площадь: от {rat.min():.4f} до {rat.max():.4f}, '
          f'разброс {(rat.max()/rat.min()-1)*100:.1f} %')
    print(f'     у наивного F_0 разброс был 19 %, показатель 0.586')
    print()
    print('СХОДИМОСТЬ ПО ДЛИНЕ БАЗИСА. Если остаток идёт от подведущих')
    print('членов, показатель обязан идти к 1/2 при их добавлении.')
    print(f"{'членов':>8}{'показатель':>13}{'ст.ош':>9}{'сигм от 1/2':>13}"
          f"{'хи²/степ профиля':>19}{'разброс отнош':>15}")
    for nb in (3, 4, 5, 6, 7):
        NB[0] = nb
        try:
            pw2 = expo(np.ones(len(A), bool))
            jk2 = np.array([expo(sd != u) for u in us])
            e2 = math.sqrt((kk - 1) / kk * ((jk2 - jk2.mean()) ** 2).sum())
            cc, chi, dof = fit_band((A[:, 0] >= q[2]) & (A[:, 0] <= q[3]))
            rr = []
            for i in range(5):
                m = (A[:, 0] >= q[i]) & (A[:, 0] <= q[i + 1])
                if m.sum() < 40:
                    continue
                rr.append(fit_band(m)[0][0]
                          / (math.pi * (A[m][:, 1] ** 2).mean() / 4.0))
            rr = np.array(rr)
            print(f'{nb:>8}{pw2:>13.4f}{e2:>9.4f}'
                  f'{abs(pw2-0.5)/max(e2,1e-9):>13.1f}'
                  f'{chi/max(dof,1):>19.2f}'
                  f'{(rr.max()/rr.min()-1)*100:>14.1f}%')
        except np.linalg.LinAlgError:
            print(f'{nb:>8}   базис вырожден')
    NB[0] = 4
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
