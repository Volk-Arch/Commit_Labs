#!/usr/bin/env python3
"""ДВА ПОТЕНЦИАЛА ПОРОЗНЬ: метрическая ли это теория вообще.

ОТКУДА ВОЗМОЖНОСТЬ. `gps.py` показал, что четыре цепи дают событию полные
координаты `(T, X)` чистым счётом, а интервал `ΔT² − |ΔX|²` совпадает с
порядковой мерой собственного времени `|I|^(1/4)`: связь +0.9896, наклон
в логарифмах +1.026, а КОЭФФИЦИЕНТ сошёлся до 0.1 % (ожидание
`1/N^(1/4) = 0.0760`, замер 0.0759). Контроль на вырождение дал ничего.

ЧТО ЭТО ОТКРЫВАЕТ. В слабом поле метрика имеет ДВА члена:

    ds² = −(1 + 2Φ) dT² + (1 − 2Ψ) dX²      ОТО требует Φ = Ψ

Мы до сих пор мерили только временной: ход часов даёт `Φ`. Пространственный
`Ψ` был недостижим, потому что одна цепь не даёт направления. Теперь
достижим, и притом ОДНОЙ подгонкой.

Порядковая мера `|I|^(1/4)` есть СОБСТВЕННЫЙ интервал (объём не зависит от
координат), а GPS даёт КООРДИНАТНЫЙ. Значит

    (|I|^(1/4) / k0)²  =  ΔT²(1 + 2Φ) − ΔX²(1 − 2Ψ)
    A − A0  =  2Φ ΔT² + 2Ψ ΔX²,        A0 = ΔT² − ΔX²
    (A − A0) / ΔT²  =  2Φ + 2Ψ v²,     v = |ΔX| / ΔT

Линейно по `v²`: **свободный член даёт 2Φ, наклон даёт 2Ψ**. Оба в одном
замере, из одного счёта.

УСТРОЙСТВО. Цепи-приборы стоят тетраэдром вокруг центра, а разрежение
сдвинуто ОТ центра на `--hd`, чтобы цепи сквозь него не проходили: прибор
не должен стоять внутри вещества. Пары событий раскладываются по
расстоянию их середины от центра разрежения, измеренному САМИМ GPS.

КОНТРОЛИ, поставленные до замера:
  1. НУЛЕВАЯ ДЫРКА обязана дать Φ = Ψ = 0 в пределах ошибки — иначе
     подгонка ловит не поле, а кривизну самого ромба;
  2. `k0` берётся из прогона БЕЗ дырки того же засева, не подгоняется;
  3. ошибка складным ножом по засевам;
  4. требуется разброс по `v²`: если все пары с одной скоростью,
     наклон и свободный член не разделяются — печатается охват по `v²`;
  5. ПЕРЕМЕШИВАНИЕ `v²` внутри полосы обязано убить наклон.

  python3 twopot.py
  python3 twopot.py --n 30000 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from gps import diamond, radar_at, trilaterate


def world(N, T, rs, rad, hc, scatter=False):
    """Ромб с разрежением вокруг точки hc (или со случайным удалением)."""
    bg = diamond(N, T, rs)
    nrem = 0
    if rad > 0:
        rr = np.linalg.norm(bg[:, 1:] - hc, axis=1)
        if scatter:
            k = int((rr < rad).sum())
            d = rs.choice(len(bg), k, replace=False)
            keep = np.ones(len(bg), bool); keep[d] = False
            nrem = k
        else:
            keep = rr >= rad
            nrem = int((~keep).sum())
        bg = bg[keep]
    return bg, nrem


def gps_coords(X, A, T, dens):
    """Полные координаты (T, X) из четырёх цепей — один счёт."""
    step = (T - 0.08) / (dens - 1)
    TT, RR, OK = [], [], []
    for i in range(4):
        t_, r_, ok_ = radar_at(X, A[i], T, dens)
        TT.append(t_); RR.append(r_); OK.append(ok_)
    TT = np.stack(TT, 1) * step
    RR = np.stack(RR, 1) * step
    ok = np.all(np.stack(OK, 1), axis=1)
    Xs, res, det = trilaterate(A, RR[ok])
    return TT[ok].mean(1), Xs, ok, res


def cards(bg, pr):
    """Мощности интервалов для пар — порядковая мера, без координат."""
    t = bg[:, 0]; S = bg[:, 1:]
    out = np.empty(len(pr))
    for q, (i, j) in enumerate(pr):
        lo, hi = t[i], t[j]
        m = (t > lo) & (t < hi)
        if not m.any():
            out[q] = 0; continue
        P = S[m]
        a1 = (t[m] - lo) ** 2 > ((P - S[i]) ** 2).sum(1)
        a2 = (hi - t[m]) ** 2 > ((S[j] - P) ** 2).sum(1)
        out[q] = int((a1 & a2).sum())
    return out


def measure(bg, A, T, dens, hc, npairs, rs, k0=None):
    """Возвращает (k0, по полосам: 2Phi, 2Psi, число пар)."""
    Tg, Xg, ok, res = gps_coords(bg, A, T, dens)
    idx = np.flatnonzero(ok)
    fin = np.isfinite(Xg).all(1)
    idx = idx[fin]; Tg = Tg[fin]; Xg = Xg[fin]
    if len(idx) < 200:
        return None
    sub = bg[idx]
    n = len(idx)
    pr = []
    tries = 0
    while len(pr) < npairs and tries < npairs * 80:
        i, j = rs.integers(0, n, 2)
        tries += 1
        if i == j:
            continue
        if sub[i, 0] > sub[j, 0]:
            i, j = j, i
        dt = sub[j, 0] - sub[i, 0]
        d2 = ((sub[j, 1:] - sub[i, 1:]) ** 2).sum()
        if dt * dt > d2:
            pr.append((i, j))
    if len(pr) < 50:
        return None
    pr = np.array(pr)
    card = cards(sub, pr)
    dT = Tg[pr[:, 1]] - Tg[pr[:, 0]]
    dX = np.linalg.norm(Xg[pr[:, 1]] - Xg[pr[:, 0]], axis=1)
    m = (card >= 4) & (dT > 0) & (dX < dT)
    if m.sum() < 40:
        return None
    pr, card, dT, dX = pr[m], card[m], dT[m], dX[m]
    sord = card ** 0.25
    if k0 is None:
        # k0 из ЭТОГО же мира (он без дырки) — не подгоняется под дырку
        A0 = dT ** 2 - dX ** 2
        k0 = float(np.sqrt((sord ** 2 / A0).mean()))
    Aq = (sord / k0) ** 2
    A0 = dT ** 2 - dX ** 2
    y = (Aq - A0) / dT ** 2
    v2 = (dX / dT) ** 2
    # середина пары по GPS — её расстояние от центра разрежения
    mid = 0.5 * (Xg[pr[:, 0]] + Xg[pr[:, 1]])
    rmid = np.linalg.norm(mid - hc, axis=1)
    return k0, y, v2, rmid, len(pr)


def main():
    ap = argparse.ArgumentParser(description='Два потенциала порознь')
    ap.add_argument('--n', type=int, default=22000)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--radell', type=float, default=2.2)
    ap.add_argument('--hd', type=float, default=0.20,
                    help='сдвиг разрежения от центра (цепи не внутри него)')
    ap.add_argument('--dens', type=int, default=460)
    ap.add_argument('--sep', type=float, default=0.13)
    ap.add_argument('--pairs', type=int, default=6000)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--bins', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    rad = a.radell * ell
    hc = np.array([a.hd, 0.0, 0.0])
    A = a.sep * np.array([[1., 1., 1.], [1., -1., -1.],
                          [-1., 1., -1.], [-1., -1., 1.]]) / math.sqrt(3)
    print(f'N = {a.n}, ell = {ell:.5f}')
    print(f'цепи тетраэдром, разнос {a.sep} = {a.sep/ell:.1f} ell')
    print(f'разрежение радиуса {rad:.4f} = {a.radell} ell в точке '
          f'({a.hd}, 0, 0) = {a.hd/ell:.1f} ell от центра')
    print(f'ближайшая цепь к центру разрежения: '
          f'{np.linalg.norm(A - hc, axis=1).min()/ell:.1f} ell '
          f'(прибор ВНЕ вещества)')
    print(f'засевов {a.seeds}')
    print()
    print('(A − A0)/ΔT² = 2Φ + 2Ψ v²  =>  свободный член Φ, наклон Ψ')
    print('ОТО требует Φ = Ψ. Если не выполнено — теория не метрическая.')
    print()

    for lbl, rr, sc in (('ДЫРКА', rad, False),
                        ('КОНТРОЛЬ рассыпано', rad, True),
                        ('КОНТРОЛЬ нулевая дырка', 0.0, False)):
        per = []
        for s in range(a.seeds):
            rsa = np.random.default_rng(31000 + 149 * s)
            bgA, _ = world(a.n, a.t, rsa, 0.0, hc)
            rr2 = np.random.default_rng(31000 + 149 * s)
            bgB, nrem = world(a.n, a.t, rr2, rr, hc, sc)
            g = np.random.default_rng(900 + s)
            RA = measure(bgA, A, a.t, a.dens, hc, a.pairs,
                         np.random.default_rng(900 + s))
            if RA is None:
                continue
            k0 = RA[0]
            RB = measure(bgB, A, a.t, a.dens, hc, a.pairs, g, k0=k0)
            if RB is None:
                continue
            per.append((RA, RB))
        if not per:
            print(f'--- {lbl}: пар не набралось'); continue
        # полосы по расстоянию середины пары от вещества
        allr = np.concatenate([p[1][3] for p in per])
        ed = np.quantile(allr, np.linspace(0, 1, a.bins + 1))
        ed[0] -= 1e-9; ed[-1] += 1e-9
        print(f'--- {lbl}')
        print(f"{'r середины':>11}{'2Φ (время)':>13}{'ст.ош':>10}"
              f"{'2Ψ (место)':>13}{'ст.ош':>10}{'Φ-Ψ сигм':>10}{'пар':>7}")
        for q in range(a.bins):
            phis, psis, npr = [], [], 0
            for RA, RB in per:
                _, y, v2, rm, _ = RB
                _, yA, v2A, rmA, _ = RA
                m = (rm >= ed[q]) & (rm < ed[q + 1])
                mA = (rmA >= ed[q]) & (rmA < ed[q + 1])
                if m.sum() < 40 or mA.sum() < 40:
                    continue
                # парно: из дырки вычитается тот же замер без дырки
                cB = np.polyfit(v2[m], y[m], 1)
                cA = np.polyfit(v2A[mA], yA[mA], 1)
                psis.append(cB[0] - cA[0])
                phis.append(cB[1] - cA[1])
                npr += int(m.sum())
            if len(phis) < 3:
                continue
            ph, ps = np.array(phis), np.array(psis)
            sp = ph.std(ddof=1) / math.sqrt(len(ph))
            ss = ps.std(ddof=1) / math.sqrt(len(ps))
            d = abs(ph.mean() - ps.mean()) / max(
                math.sqrt(sp ** 2 + ss ** 2), 1e-12)
            c = 0.5 * (ed[q] + ed[q + 1])
            print(f'{c/ell:>11.2f}{ph.mean():>+13.5f}{sp:>10.5f}'
                  f'{ps.mean():>+13.5f}{ss:>10.5f}{d:>10.1f}{npr:>7}')
            # РАЗЛИЧЕНИЕ ДВУХ ГИПОТЕЗ, выведенное, а не подогнанное.
            #
            # Сдвиг ПЛОТНОСТИ: удаление событий меняет rho, а |I|^(1/4)
            # от rho зависит, значит A -> cA целиком, и тогда
            #     (A − A0)/ΔT² = (c−1)(1 − v²)
            # то есть свободный член = МИНУС наклон.
            #
            # Метрика ОТО (Φ = Ψ): свободный член = ПЛЮС наклон.
            #
            # Две гипотезы отличаются знаком, поэтому разделяются сами.
            se2 = math.sqrt(sp ** 2 + ss ** 2)
            dens_h = abs(ph.mean() + ps.mean()) / max(se2, 1e-12)
            gr_h = abs(ph.mean() - ps.mean()) / max(se2, 1e-12)
            verdict = ('ПЛОТНОСТЬ' if dens_h < 2 and gr_h > 3 else
                       'метрика ОТО' if gr_h < 2 and dens_h > 3 else
                       'ни одна' if dens_h > 3 and gr_h > 3 else 'не решено')
            print(f'{"":>11}  гипотеза плотности |Φ+Ψ| = {dens_h:.1f} сигм, '
                  f'гипотеза ОТО |Φ−Ψ| = {gr_h:.1f} сигм  ->  {verdict}')
        v2all = np.concatenate([p[1][2] for p in per])
        print(f'  охват по v²: от {np.quantile(v2all,0.05):.3f} до '
              f'{np.quantile(v2all,0.95):.3f} — '
              f'{"разделение возможно" if np.quantile(v2all,0.95)-np.quantile(v2all,0.05) > 0.2 else "РАЗБРОСА МАЛО"}')
        print(flush=True)

    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
