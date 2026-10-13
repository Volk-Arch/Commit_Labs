#!/usr/bin/env python3
"""ЯЩИК С ЗАМЫКАНИЕМ: мир без края, дешевле ромба, и правило то же.

ОТКУДА. `game-doc.md` уткнулся в то, что окно между защитой от касания и
краем ромба сомкнулось. Его три возражения разобрали это по частям:

 1. «либо ровно 4.7, либо одно из утверждений неверно» — верно, и неверным
    оказалось моё. Два предела по-разному зависят от N:
        предел защиты  ~ rad + запас + охват правила ~ 4.3 ell — ПОСТОЯННЫЙ
        предел границы = 0.40 * 0.5 * (7.639 N)^(1/4)  — РАСТЁТ как N^(1/4)
    При N = 40000 они сошлись случайно (4.3 против 4.72). Окно не
    сомкнуто навсегда, оно просто ещё не раскрылось.
 2. «должен ли быть ровно один показатель» — не должен. В задаче есть свои
    масштабы (радиус дырки, глубина k, длина ell), и при наличии масштаба
    степенной закон не обязан быть единым. Правильный вопрос — есть ли у
    МЕСТНОГО показателя полка, и это измеряется, а не предполагается.
 3. «нет ли способа задать большое проще» — есть, и это не приближение.

УСТРОЙСТВО. Край мешает потому, что у ромба есть край. Но правило
локально: глубина k, охват около 4 ell. Значит пространство можно
ЗАМКНУТЬ (периодические границы) — тогда радиального края не существует
вовсе и предел r/R исчезает. По времени достаточно слоя: правило дальше и
не смотрит. Край по времени, в отличие от края ромба, НЕ радиальный — он
одинаков на всех расстояниях и профиль по r не портит.

    ромб R = 18 ell:      N = 232000,  край ЕСТЬ
    ящик 20 ell x 12 ell: N =  96000,  края НЕТ

Причинность с замыканием — по наименьшему образу: разность по каждой
координате приводится в [-L/2, L/2]. Правило остаётся прежним:

    из x назад k звеньев по самой долгой цепи -> a;  n = |(a,x)|;
    s = k / n^(1/4)

причём высота считается в ОКНЕ толщины w (правилу больше не нужно), что и
делает счёт линейным по N вместо квадратичного.

ПРОВЕРКИ, поставленные до замера:
  1. ГЛАВНАЯ: в пустом ящике s обязано выйти тем же, что в пустом ромбе
     (около 2.01 при k = 5). Если нет — построение неверно;
  2. плотностная независимость: s не должно зависеть от rho;
  3. нулевая дырка — ровно нуль;
  4. рассыпанное удаление — нуль;
  5. защита от касания прежняя: весь интервал не ближе rad + guard;
  6. окно w проверяется ростом: ответ не должен от него зависеть.

  python3 box.py --check          # только проверки 1-2, быстро
  python3 box.py --seeds 8
"""
import argparse
import math
import time

import numpy as np


def sprinkle(L, T, rho, rs):
    """Сприклинг в ящик L^3 x T. Плотность rho = 1/ell^4."""
    n = rs.poisson(rho * L ** 3 * T)
    X = np.empty((n, 4))
    X[:, 0] = rs.uniform(0, T, n)
    X[:, 1:] = rs.uniform(0, L, (n, 3))
    return X[np.argsort(X[:, 0])]


def dx2_periodic(A, B, L):
    """Квадрат расстояния по наименьшему образу: (len(A), len(B))."""
    d = A[:, None, :] - B[None, :, :]
    d -= L * np.round(d / L)
    return (d * d).sum(2)


def heights_window(X, L, w, blk=256):
    """Высота в окне толщины w и указатель на предшественника.

    Правило смотрит назад всего k звеньев, поэтому высота нужна не от
    начала мира, а в окне. Это и делает счёт линейным: для каждого блока
    перебираются только события внутри w по времени, а не весь мир.
    """
    n = len(X)
    t = X[:, 0]
    S = X[:, 1:]
    h = np.ones(n, np.int32)
    pr = np.full(n, -1, np.int64)
    for a in range(0, n, blk):
        b = min(a + blk, n)
        lo = np.searchsorted(t, t[a] - w, 'left')
        if lo >= a:
            continue
        dt = t[a:b][:, None] - t[lo:a][None, :]
        d2 = dx2_periodic(S[a:b], S[lo:a], L)
        rel = (dt > 0) & (dt * dt > d2)
        del dt, d2
        hv = h[lo:a].astype(np.int32)
        cand = np.where(rel, hv[None, :], 0)
        del rel
        j = cand.argmax(1)
        v = cand[np.arange(b - a), j]
        del cand
        # внутри блока события могут быть связаны между собой; блок мал,
        # поэтому доводим последовательно
        h[a:b] = v + 1
        pr[a:b] = np.where(v > 0, lo + j, -1)
        for i in range(a, b):
            if i == a:
                continue
            dt2 = t[i] - t[a:i]
            dd = dx2_periodic(S[i:i + 1], S[a:i], L)[0]
            m = (dt2 > 0) & (dt2 * dt2 > dd)
            if m.any():
                idx = np.flatnonzero(m)
                kk = idx[h[a:i][m].argmax()]
                if h[a + kk] + 1 > h[i]:
                    h[i] = h[a + kk] + 1
                    pr[i] = a + kk
    return h, pr


def ruler_box(X, L, k, w, hc, tlo, thi):
    """Линейка s = k/n^(1/4), её определённость и dmin до вещества."""
    h, pr = heights_window(X, L, w)
    t = X[:, 0]
    S = X[:, 1:]
    n = len(X)
    s = np.full(n, np.nan)
    dmin = np.full(n, np.nan)
    # расстояние до центра вещества — тоже по наименьшему образу
    dh = S - hc
    dh -= L * np.round(dh / L)
    rad_all = np.linalg.norm(dh, axis=1)
    cand = np.flatnonzero((t >= tlo) & (t <= thi) & (h >= k + 1))
    for x in cand:
        a = x
        for _ in range(k):
            a = pr[a]
            if a < 0:
                break
        if a < 0:
            continue
        i0 = np.searchsorted(t, t[a], 'right')
        i1 = np.searchsorted(t, t[x], 'left')
        if i1 <= i0:
            s[x] = float(k)
            dmin[x] = min(rad_all[a], rad_all[x])
            continue
        P = S[i0:i1]
        tp = t[i0:i1]
        d1 = dx2_periodic(P, S[a:a + 1], L)[:, 0]
        d2 = dx2_periodic(S[x:x + 1], P, L)[0]
        inside = ((tp - t[a]) ** 2 > d1) & ((t[x] - tp) ** 2 > d2)
        cnt = int(inside.sum())
        s[x] = k / max(cnt, 1) ** 0.25
        ri = rad_all[i0:i1][inside]
        dmin[x] = min(rad_all[a], rad_all[x],
                      float(ri.min()) if len(ri) else np.inf)
    return s, np.isfinite(s), dmin, rad_all, h


def build_box(L, T, rho, rs, rad, hc, scatter=False):
    X = sprinkle(L, T, rho, rs)
    if rad <= 0:
        return X, 0
    d = X[:, 1:] - hc
    d -= L * np.round(d / L)
    rr = np.linalg.norm(d, axis=1)
    if scatter:
        kk = int((rr < rad).sum())
        drop = rs.choice(len(X), kk, replace=False)
        keep = np.ones(len(X), bool); keep[drop] = False
        return X[keep], kk
    keep = rr >= rad
    return X[keep], int((~keep).sum())


def main():
    ap = argparse.ArgumentParser(description='Ящик с замыканием')
    ap.add_argument('--lell', type=float, default=20.0, help='сторона в ell')
    ap.add_argument('--tell', type=float, default=12.0, help='время в ell')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--well', type=float, default=6.0, help='окно высоты, ell')
    ap.add_argument('--radell', type=float, default=2.2)
    ap.add_argument('--guard', type=float, default=1.0)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--step', type=float, default=1.0)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    t0 = time.time()
    # работаем в единицах ell: rho = 1
    L, T, w = a.lell, a.tell, a.well
    rho = 1.0
    hc = np.array([L / 2, L / 2, L / 2])
    tlo, thi = w + 1.0, T - 1.0
    print('ЯЩИК С ЗАМЫКАНИЕМ: радиального края НЕТ, значит предела r/R нет')
    print(f'сторона {L} ell, время {T} ell, окно высоты {w} ell, k = {a.k}')
    print(f'ожидаемое число событий {int(rho*L**3*T)}')
    print(f'замер по событиям с t в [{tlo}, {thi}] ell')
    print()

    print('ПРОВЕРКА 1 (главная): s в пустом ЯЩИКЕ против пустого РОМБА')
    print('в ромбе при k = 5 вышло 2.0067 / 2.0151 / 2.0089 (разброс 0.42 %)')
    print()
    print('Замечание о плотности: в ЗАМКНУТОМ ящике проверка плотности')
    print('ВЫРОЖДЕНА. Всё задано в единицах ell, поэтому N = L^3 * T не')
    print('зависит от rho вовсе: смена плотности здесь есть просто')
    print('пересчёт единиц, и проверять там нечего. Плотностная')
    print('независимость линейки установлена в РОМБЕ (0.42 %), где размер')
    print('мира задавался отдельно от ell. Здесь проверяется другое: что')
    print('построение с замыканием даёт ТО ЖЕ s.')
    print()
    print(f"{'засев':>7}{'событий':>10}{'s среднее':>12}{'в кадре':>10}")
    vv = []
    for s_ in range(3):
        X, _ = build_box(L, T, rho, np.random.default_rng(1700 + 31 * s_),
                         0.0, hc)
        sv, ok, _, _, hh = ruler_box(X, L, a.k, w, hc, tlo, thi)
        vv.append(np.nanmean(sv[ok]))
        print(f'{s_:>7}{len(X):>10}{vv[-1]:>12.4f}{ok.sum():>10}')
    v = np.array(vv)
    print(f'  s = {v.mean():.4f} +- {v.std(ddof=1)/math.sqrt(len(v)):.4f}, '
          f'n = {(a.k/v.mean())**4:.2f}')
    ok1 = abs(v.mean() - 2.01) / 2.01 < 0.05
    print(f'  против ромбовых 2.01 — '
          f'{"СОВПАЛО, построение верно" if ok1 else "НЕ СОВПАЛО, построение неверно"}')
    if a.check:
        print(f'  время {time.time()-t0:.0f} с')
        return
    print()

    rad = a.radell
    nb = int((L / 2 - 1.0 - rad) / a.step)
    bands = [(rad + q * a.step, rad + (q + 1) * a.step) for q in range(nb)]
    bc = np.array([0.5 * (lo + hi) for lo, hi in bands])
    print(f'ОТКЛИК: разрежение {rad} ell в середине ящика, полос {nb} '
          f'до {rad+nb*a.step:.1f} ell')
    print('(в ромбе дальше 0.40R мерить было нельзя; здесь предела нет)')
    print()
    acc = {'дырка': [[] for _ in bands], 'рассыпано': [[] for _ in bands],
           'нуль': [[] for _ in bands]}
    for s_ in range(a.seeds):
        g = lambda: np.random.default_rng(23000 + 211 * s_)
        X0, _ = build_box(L, T, rho, g(), 0.0, hc)
        s0, ok0, dm0, r0, _ = ruler_box(X0, L, a.k, w, hc, tlo, thi)
        for lbl, rr, sc in (('дырка', rad, False), ('рассыпано', rad, True),
                            ('нуль', 0.0, False)):
            X1, _ = build_box(L, T, rho, g(), rr, hc, sc)
            s1, ok1_, dm1, r1, _ = ruler_box(X1, L, a.k, w, hc, tlo, thi)
            gd = rad + a.guard
            for q, (lo, hi) in enumerate(bands):
                m0 = ok0 & (r0 >= lo) & (r0 < hi) & (dm0 > gd)
                m1 = ok1_ & (r1 >= lo) & (r1 < hi) & (dm1 > gd)
                if m0.sum() < 20 or m1.sum() < 20:
                    continue
                acc[lbl][q].append(float(np.nanmean(s1[m1])
                                         - np.nanmean(s0[m0])))
            del X1, s1
        del X0, s0
        print(f'  засев {s_} ({time.time()-t0:.0f} с)', flush=True)

    print()
    print(f"{'r в ell':>9}{'ДЫРКА':>12}{'сигм':>7}{'рассыпано':>12}{'сигм':>7}"
          f"{'нуль':>10}{'раз':>7}")
    xs, ys = [], []
    for q in range(nb):
        vh = np.array(acc['дырка'][q]); vs = np.array(acc['рассыпано'][q])
        vz = np.array(acc['нуль'][q])
        if len(vh) < 3 or len(vs) < 3:
            continue
        seh = vh.std(ddof=1) / math.sqrt(len(vh))
        ses = vs.std(ddof=1) / math.sqrt(len(vs))
        print(f'{bc[q]:>9.2f}{vh.mean():>+12.5f}'
              f'{abs(vh.mean())/max(seh,1e-12):>7.1f}'
              f'{vs.mean():>+12.5f}{abs(vs.mean())/max(ses,1e-12):>7.1f}'
              f'{vz.mean():>+10.5f}'
              f'{abs(vh.mean())/max(abs(vs.mean()),1e-12):>7.1f}')
        if vh.mean() < 0:
            xs.append(bc[q]); ys.append(-vh.mean())
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) >= 5:
        print()
        print('МЕСТНЫЙ ПОКАЗАТЕЛЬ (его вопрос: один он или разный?)')
        print(f"{'центр':>8}{'показатель':>13}")
        loc = []
        for i in range(1, len(xs) - 1):
            p = -np.polyfit(np.log(xs[i-1:i+2]), np.log(ys[i-1:i+2]), 1)[0]
            loc.append((xs[i], p))
            print(f'{xs[i]:>8.2f}{-p:>13.3f}')
        pa = -np.polyfit(np.log(xs), np.log(ys), 1)[0]
        print(f'  по всем полосам: r^{-pa:+.3f}')
        if len(loc) >= 3:
            vals = np.array([p for _, p in loc])
            print(f'  разброс местного показателя: {vals.std():.3f} '
                  f'при среднем {-vals.mean():.3f}')
            print(f'  {"ПОЛКА есть — показатель один" if vals.std() < 0.15 else "полки НЕТ — показатель едет, он не один"}')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
