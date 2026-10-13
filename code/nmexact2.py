#!/usr/bin/env python3
"""ДВУХТОЧЕЧНЫЙ ИНТЕГРАЛ, исправленный, и ответ на ставку.

Что исправлено против nmexact.py:

1. Доля связанных печаталась с лишним множителем 2 (замер был верен:
   0.10005 против точного 0.1).

2. Постоянная `a` в законе q(tau) -> a*tau подгонялась по окну с
   tau до 0.3, где q/tau уже загнулось (4.15 -> 1.17). Взвешенное
   среднее село на большие tau, вышло a = 1.21, и я напечатал
   «плоско, закон подтверждён» при наклоне -0.31. Теперь `a` берётся
   как предел: q/tau = a + b*tau + c*tau^2 по окну малых tau, и
   проверяется НЕЗАВИСИМО — асимптотическая постоянная (a/8)*sum g
   обязана совпасть с пределом P/N^1.5 прямого оценщика при N -> 10^8.
   Это контроль, который ловит именно такую ошибку.

3. Главное по существу. Разброс предсказывает НЕ `P`, а взвешенная
   сумма `Q = sum_m w_m^2 <N_m>`, потому что смесь меняется с N
   (N1/N0 растёт 0.209 -> 0.492 к асимптотическому 0.5), а веса
   сидят на N1 и N2 (81 и 256). Поэтому `Q` растёт БЫСТРЕЕ `P`, и
   показатель sigma равен половине показателя `Q`, а не `P`. Отсюда
   и расхождение 0.13, помеченное в sigma-doc.md §4.

ПРОВЕРКА ЭТОГО. Половина местного показателя `Q` в окне N = 200…3200
обязана совпасть с замеренным показателем sigma = 0.974 ± 0.030
(`sigshot.py`). Если совпадёт — закон замкнулся и §4 снимается.

СТАВКА. Слоистый сектор имеет показатель ровно 1. Многообразный даёт
половину показателя `Q`. Где разрыв превысит 2 сигмы при типичной
ошибке замера 0.03 — там и смотреть.
"""
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)
WT = np.array([-1.0, 9.0, -16.0, 8.0])
MFAC = np.array([1.0, 1.0, 2.0, 6.0])
GAM = np.array([math.gamma(m + 0.5) for m in range(4)]) / MFAC


def uniform_diamond(m, rs):
    out, got = [], 0
    while got < m:
        k = int((m - got) * 5) + 4096
        t = rs.random(k)
        v = rs.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(k) ** (1.0 / 3.0)) * 0.5
        ok = rad < np.minimum(t, 1 - t)
        if ok.any():
            out.append(np.column_stack([t[ok], (v * rad[:, None])[ok]]))
            got += int(ok.sum())
    return np.vstack(out)[:m]


def taus(npair, rs):
    A = uniform_diamond(npair, rs)
    B = uniform_diamond(npair, rs)
    dt = np.abs(B[:, 0] - A[:, 0])
    dr = np.linalg.norm(B[:, 1:] - A[:, 1:], axis=1)
    t2 = (dt - dr) * (dt + dr)
    return np.sqrt(t2[t2 > 0])


def nm(tau, M, Ns):
    out = np.zeros((len(Ns), 4))
    for i, N in enumerate(Ns):
        lam = N * tau ** 4
        e = np.exp(-lam)
        for m in range(4):
            out[i, m] = (N * N / 2.0) * (e * lam ** m / MFAC[m]).sum() / M
    return out


def loc_exp(x, y):
    return np.diff(np.log(y)) / np.diff(np.log(x))


def main():
    print(__doc__)
    rs = np.random.default_rng(2718281)
    CH, NCH = 2_500_000, 24
    parts, M = [], 0
    for _ in range(NCH):
        parts.append(taus(CH, rs))
        M += CH
    tau = np.concatenate(parts)
    print(f'собрано {M:,} пар, связанных {len(tau):,}')
    r = len(tau) / M
    print(f'КОНТРОЛЬ A: доля связанных {r:.5f} ± '
          f'{math.sqrt(len(tau))/M:.5f} против точного 0.1 '
          f'({abs(r-0.1)/(math.sqrt(len(tau))/M):.1f} сигм)')

    # --- a как ПРЕДЕЛ, а не как среднее по окну ---
    edges = np.logspace(-2.6, -0.5, 43)
    cnt, _ = np.histogram(tau, bins=edges)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    qt = cnt / M / np.diff(edges) / ctr
    er = qt / np.sqrt(np.maximum(cnt, 1))
    print('\nПОСТОЯННАЯ a как предел q/tau при tau -> 0')
    for hi, deg in ((0.05, 1), (0.10, 2), (0.20, 2), (0.20, 3)):
        s = (ctr < hi) & (cnt > 200)
        c = np.polyfit(ctr[s], qt[s], deg, w=1 / er[s])
        res = (qt[s] - np.polyval(c, ctr[s])) / er[s]
        print(f'  окно tau < {hi}, степень {deg}: a = {c[-1]:.4f}, '
              f'хи2/d.o.f. = {float((res**2).sum()/(s.sum()-deg-1)):.2f} '
              f'({int(s.sum())} полос)')
    s = (ctr < 0.10) & (cnt > 200)
    c = np.polyfit(ctr[s], qt[s], 2, w=1 / er[s])
    a = float(c[-1])
    A_m = (a / 8.0) * GAM
    print(f'  принято a = {a:.4f}  =>  A_m = '
          + ', '.join(f'{v:.4f}' for v in A_m)
          + f', sum = {A_m.sum():.4f}')

    # --- большой диапазон ---
    grid = np.array([10 ** (2 + 0.125 * i) for i in range(57)])
    V = nm(tau, M, grid)
    P = V.sum(axis=1)
    Q = (V * WT ** 2).sum(axis=1)

    print('\nКОНТРОЛЬ E (ловит испорченную a): асимптота из a против '
          'предела прямого оценщика')
    print(f'  (a/8) sum Gamma = {A_m.sum():.4f}')
    print(f'  P/N^1.5 при N = 10^7, 10^8, 10^9: '
          + ', '.join(f'{P[i]/grid[i]**1.5:.4f}'
                      for i in (40, 48, 56)))
    print(f'  расхождение на N=10^9: '
          f'{abs(P[56]/grid[56]**1.5 - A_m.sum())/A_m.sum()*100:.1f}% '
          f'(прежняя a = 1.21 дала бы 0.587 против ~2.0, то есть '
          f'в 3.4 раза мимо — вот чем ловится)')

    print('\n=== ОТВЕТ НА §4: половина показателя Q против замера ===')
    w = (grid >= 200) & (grid <= 3200)
    iw = np.where(w)[0]
    eQ = (math.log(Q[iw[-1]] / Q[iw[0]])
          / math.log(grid[iw[-1]] / grid[iw[0]]))
    eP = (math.log(P[iw[-1]] / P[iw[0]])
          / math.log(grid[iw[-1]] / grid[iw[0]]))
    print(f'  в окне N = 200…3200:  показатель P = {eP:.4f} '
          f'(половина {eP/2:.4f}),  показатель Q = {eQ:.4f} '
          f'(половина {eQ/2:.4f})')
    print(f'  замерено sigma (sigshot.py):  0.9741 ± 0.0295')
    print(f'  от половины P:  {abs(0.9741-eP/2)/0.0295:.1f} сигм   '
          f'<-- расхождение §4')
    print(f'  от половины Q:  {abs(0.9741-eQ/2)/0.0295:.1f} сигм   '
          f'<-- с правильной величиной')

    print('\n=== СТАВКА: где секторы расходятся ===')
    print(f"{'N':>10}{'пок. P':>9}{'пок. Q':>9}"
          f"{'пред. sigma':>13}{'разрыв с 1':>12}{'сигм при 0.03':>15}")
    eQl = loc_exp(grid, Q)
    ePl = loc_exp(grid, P)
    for i in range(0, len(eQl), 4):
        pred = eQl[i] / 2.0
        gap = 1.0 - pred
        print(f'{grid[i+1]:>10.3g}{ePl[i]:>9.4f}{eQl[i]:>9.4f}'
              f'{pred:>13.4f}{gap:>12.4f}'
              f'{gap/math.sqrt(2)/0.03:>15.1f}')
    for s in (2.0, 3.0, 5.0):
        hit = next((grid[i + 1] for i in range(len(eQl))
                    if (1.0 - eQl[i] / 2) / math.sqrt(2) / 0.03 >= s), None)
        print(f'  разрыв >= {s:.0f} сигм: N ~ '
              + (f'{hit:.3g}' if hit else 'не достигнут'))

    print('\nасимптотические пределы (для записи)')
    print(f'  P -> {A_m.sum():.4f} N^1.5,  '
          f'Q -> {float((WT**2) @ A_m):.3f} N^1.5,  '
          f'sigma_шум -> {K4*math.sqrt(float((WT**2) @ A_m)):.3f} N^0.75')
    print(f'  смесь N_m/N_0 при N=10^9: '
          + ', '.join(f'{V[56,m]/V[56,0]:.4f}' for m in range(4))
          + '  против точных 1, 0.5, 0.375, 0.3125')


if __name__ == '__main__':
    main()
