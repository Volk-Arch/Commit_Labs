#!/usr/bin/env python3
"""СВЯЗАННЫЙ АНСАМБЛЬ: три объекта в одном прогоне, на одних засевах.

ВОПРОС. Архив (`weight-doc.md`) намерил в связанном ансамбле `exp(−βS)`
с закреплённым числом связанных пар: **6.1σ обеднение** фона внутри 0.27
от оси ТРУБКИ, подъём на 0.38–0.5. Фон выдавливается от объекта.
Истолковано как болезнь конформного множителя евклидовой квантовой
гравитации (`exp(−S)` предпочитает антигравитирующие конфигурации).

Но вчера установлено (`delay-doc.md`): плотная трубка — не масса, а
ЛЕСТНИЦА (часы быстрее, цепи загибаются внутрь — обе половины тяготения
с обратным знаком). Масса есть РАЗРЕЖЕНИЕ (часы медленнее 4.5σ, цепи
наружу 5.0σ). Значит ансамбль просил разрежения на месте трубки — то есть
просил МАССУ, — а это записали как болезнь.

ПРЕДСКАЗАНИЕ, зарегистрированное ДО замера:
    трубка    -> обеднение у объекта (воспроизведение архива)
    разрежение -> СБОР у объекта, то есть знак перевёрнут

Исходы:
  трубка обедняет + разрежение собирает -> гипотеза верна
  обе обедняют                          -> гипотеза неверна, диагноз
                                           архива (конформный множитель)
                                           остаётся
  трубка не обедняет                    -> машинерия не архивная, и оба
                                           вывода пусты

ПОЧЕМУ ЭТО НЕ ПАДАЕТ ПОД СТЕНУ §15. Стена (`bdwall-doc.md`) стоит на `b`
и `B` как ИЗМЕРИТЕЛЯХ кривизны: сигнал `ℓ²R`, шум `N^0.357`, нужно ~10⁹
событий. Здесь действие работает ВЕСОМ: сигнал — не значение `S`, а куда
ансамбль расставил события.

ДВЕ ЛОВУШКИ, обойдённые заранее.
 1. Край. В пробе корзина 0.35–0.40 дала 1.112 ± 0.024 (4.7σ) при объекте
    в ЦЕНТРЕ — то есть ансамбль растаскивает события относительно
    светового края, где члены действия огромны (⟨b⟩ = −0.611 при 24.5σ на
    высотах 1–2). Поэтому отсчёт берётся от прогона БЕЗ объекта, и
    краевое вычитается.
 2. Рабочая точка. При β = 1 принимается 3.4 % — за 30000 шагов это 3.6
    принятых хода на событие, до равновесия далеко. Архив опирался на
    27 %. Подбор дал: при N = 220 β = 0.1 даёт **26.8 %**, то есть
    архивная рабочая точка, и заодно показал, что нормировки действия
    различаются примерно в 10 раз.

  python3 ens2.py
  python3 ens2.py --n 220 --betas 0 0.1 --steps 60000 --seeds 4
"""
import argparse
import math
import time

import numpy as np

from ensemble import inside_diamond, allowed

W = (-1.0, 9.0, -16.0, 8.0)


def action_and_pairs(X):
    """Полное действие БД и число связанных пар. БЫСТРАЯ версия.

    УСКОРЕНИЕ, найденное профилированием. Узким местом была матрица
    расстояний через broadcasting — 2.484 мс из 2.90 мс на шаг. Через
    матрицу Грама `d² = g_i + g_j − 2·S·Sᵀ` это 0.232 мс, то есть
    в 10.7 раза быстрее; плюс `bincount` вместо четырёх проходов
    (0.089 против 0.135 мс); плюс сравнение КВАДРАТОВ вместо корня —
    и точнее, и быстрее. Итого шаг 2.90 -> 0.60 мс.

    Сверено с медленной версией на 40 случайных мирах: расхождений 0.
    """
    t = X[:, 0]
    dt = t[None, :] - t[:, None]
    S = X[:, 1:]
    g = (S * S).sum(1)
    d2 = g[:, None] + g[None, :] - 2.0 * (S @ S.T)
    np.maximum(d2, 0.0, out=d2)
    R = (dt > 0) & (dt * dt > d2)
    A = R.astype(np.float32)
    c = np.bincount((A @ A)[R].astype(np.int64), minlength=4)
    s = float(len(X))
    for m in range(4):
        s += W[m] * float(c[m])
    return s, int(R.sum())


def sprinkle(N, T, rs, rad, cen):
    out, got = [], 0
    while got < N:
        m = (N - got) * 9 + 128
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        Y = Y[inside_diamond(Y, T)]
        if len(Y) and rad > 0:
            ok = np.ones(len(Y), bool)
            for c in cen:
                ok &= np.linalg.norm(Y[:, 1:] - c[None, :], axis=1) >= rad
            Y = Y[ok]
        if len(Y):
            out.append(Y)
            got += len(Y)
    return np.vstack(out)[:N]


def run(N, T, beta, steps, rad, cen, M, tol, rs, jump):
    mob = sprinkle(N, T, rs, rad, cen)
    if M > 0:
        frz = np.zeros((M, 4))
        frz[:, 0] = np.linspace(0.12, T - 0.12, M)
        X = np.vstack([mob, frz])
    else:
        X = mob.copy()
    nm = len(mob)
    S, P0 = action_and_pairs(X)
    acc, tried, prof, nsamp, pairs = 0, 0, np.zeros(10), 0, []
    edges = np.linspace(0.0, 0.5, 11)
    burn = steps // 3
    for t in range(steps):
        i = int(rs.integers(0, nm))
        old = X[i].copy()
        p = old.copy()
        p[0] += rs.normal(0, jump)
        p[1:] += rs.normal(0, jump, 3)
        if not allowed(p, rad, cen, T):
            continue
        X[i] = p
        Sn, Pn = action_and_pairs(X)
        tried += 1
        if abs(Pn - P0) > tol * P0:
            X[i] = old
            continue
        d = Sn - S
        if d <= 0 or rs.random() < math.exp(-beta * d):
            S = Sn
            acc += 1
        else:
            X[i] = old
        if t >= burn and t % 30 == 0:
            r = np.linalg.norm(X[:nm, 1:], axis=1)
            prof += np.histogram(r, bins=edges)[0]
            nsamp += 1
            pairs.append(Pn)
    return (prof / max(nsamp, 1), acc / max(tried, 1),
            np.array(pairs))


def main():
    ap = argparse.ArgumentParser(description='Ансамбль: три объекта')
    ap.add_argument('--n', type=int, default=220)
    ap.add_argument('--rad', type=float, default=0.16)
    ap.add_argument('--tube', type=int, default=26)
    ap.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.1])
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--tol', type=float, default=0.004)
    ap.add_argument('--jump', type=float, default=0.06)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    print(f'N подвижных = {a.n}, ромб T = 1 (радиус 0.5), ell = {ell:.4f}')
    print(f'разрежение: запретный шар радиуса {a.rad} = {a.rad/ell:.1f} ell')
    print(f'трубка: {a.tube} замороженных событий на оси')
    print(f'закрепление пар {100*a.tol:.1f} %, шагов {a.steps}, '
          f'засевов {a.seeds}')
    print()
    print('ПРЕДСКАЗАНИЕ, зарегистрированное ДО замера:')
    print('  трубка (анти-масса) -> обеднение у оси  (архив: 6.1σ)')
    print('  разрежение (масса)  -> СБОР у объекта, знак перевёрнут')
    print()

    cfg = [('ничего', 0.0, [], 0),
           ('разрежение', a.rad, [np.zeros(3)], 0),
           ('трубка', 0.0, [], a.tube)]
    store = {}
    for nm, rad, cen, M in cfg:
        print(f'--- {nm}')
        base = {}
        for beta in a.betas:
            P, AC, PR = [], [], []
            for s in range(a.seeds):
                pr, ac, pp = run(a.n, 1.0, beta, a.steps, rad, cen, M,
                                 a.tol, np.random.default_rng(
                                     23300 + 101 * s), a.jump)
                P.append(pr); AC.append(ac)
                if len(pp):
                    PR.append(pp.std() / max(pp.mean(), 1))
            base[beta] = np.array(P)
            print(f'  beta = {beta:<5.2f} принято {100*np.mean(AC):5.1f} %'
                  f'   разброс пар {100*np.mean(PR):.2f} %', flush=True)
        store[nm] = base

    ed = np.linspace(0.0, 0.5, 11)
    for beta in [b for b in a.betas if b != 0.0]:
        print()
        print(f'Отношение плотности фона beta={beta:g} к beta=0, '
              f'и РАЗНИЦА с «ничего»')
        print(f"  {'радиус':>12}{'ничего':>10}{'разрежение':>13}"
              f"{'разн.':>9}{'сигм':>6}{'трубка':>11}{'разн.':>9}"
              f"{'сигм':>6}")
        for q in range(10):
            ok, val = True, {}
            for nm in ('ничего', 'разрежение', 'трубка'):
                b0 = store[nm][0.0][:, q]
                if b0.mean() < 3:
                    ok = False
                    break
                val[nm] = store[nm][beta][:, q] / max(b0.mean(), 1e-9)
            if not ok:
                continue
            u = val['ничего']
            su = u.std(ddof=1) / math.sqrt(len(u))
            line = f'  {f"{ed[q]:.2f}-{ed[q+1]:.2f}":>12}{u.mean():>10.3f}'
            for nm in ('разрежение', 'трубка'):
                v = val[nm]
                d = v.mean() - u.mean()
                sd = math.hypot(su, v.std(ddof=1) / math.sqrt(len(v)))
                line += (f'{v.mean():>13.3f}{d:>+9.3f}'
                         f'{abs(d)/max(sd,1e-9):>6.1f}')
            print(line)

    print()
    print('  трубка < 0 у оси И разрежение > 0 => знак перевёрнут,')
    print('  гипотеза верна; обе < 0 => диагноз архива стоит;')
    print('  трубка ~ 0 => машинерия не архивная, выводы пусты')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
