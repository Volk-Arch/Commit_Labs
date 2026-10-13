#!/usr/bin/env python3
"""СКОЛЬКО КОГЕРЕНТНОЙ ФАЗЫ ДАЮТ ЧАСЫ ПОРЯДКА.

Три замера (6 окт., утро 10 окт., час назад) мерили фазу
ГРАВИТАЦИОННОГО действия и дали отношение сигнала к разбросу
mu/sigma <= 0.29, то есть 1/20 радиана когерентной фазы. Причина
названа: в плоском мире классическое гравитационное действие равно
нулю, а шум растёт как сигнал (sigma ~ N^0.967).

Но фаза квантовой механики — это exp(i m tau / hbar), и собственное
время проект читает ИЗ ПОРЯДКА. Вопрос: с какой точностью, и сколько
радиан когерентной фазы это даёт.

ПРОГНОЗ, сделанный ДО прогона. Объём диаманта V = (pi/24) tau^4 при
d = 4, значит |I| ~ Poisson(rho V) и оценка tau = (24|I|/(pi rho))^{1/4}
имеет относительную ошибку (1/4)/sqrt(|I|). Отсюда

    когерентная фаза  Phi_max = 4 sqrt(N) радиан

то есть часы с размером УЛУЧШАЮТСЯ. У действия отношение было
постоянным; это первое место, где размер работает за нас.

КОНТРОЛИ:
  1. связь |I| с tau обязана дать степень 4 и коэффициент (pi/24) rho
     — это проверка засева и оценщика, а не подгонка;
  2. погрешности берутся по НЕЗАВИСИМЫМ засевам, а не по парам внутри
     одного засева (пары коррелированы общим полем);
  3. при искусственно ЗАВЫШЕННОЙ ошибке когерентность обязана умирать
     раньше — проверка того, что мерится именно ошибка оценки;
  4. прямой замер |<exp(i Phi)>| обязан совпасть с exp(-Phi^2 s^2/2)
     при малых s (гауссов предел).

  python3 qclock.py
"""
import math

import numpy as np


def sprinkle_diamond(n, rs, T=1.0):
    pts = []
    while len(pts) < n:
        m = int((n - len(pts)) * 5) + 64
        t = rs.random(m) * T
        v = rs.normal(size=(m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(m) ** (1.0 / 3.0)) * (T / 2)
        x = v * rad[:, None]
        ok = rad < np.minimum(t, T - t)
        for i in np.nonzero(ok)[0]:
            pts.append(np.concatenate([[t[i]], x[i]]))
    return np.array(pts[:n])


def rel_matrix(P):
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt ** 2 > dx2), dt, dx2


def sample_pairs(P, Rc, dt, dx2, npair, rs):
    """Случайные связанные пары: tau_true и |I|."""
    n = len(P)
    xs, ys = np.nonzero(Rc)
    if len(xs) == 0:
        return np.empty(0), np.empty(0)
    idx = rs.choice(len(xs), size=min(npair, len(xs)), replace=False)
    out_tau, out_I = [], []
    for i in idx:
        x, y = int(xs[i]), int(ys[i])
        tau = math.sqrt(max(dt[x, y] ** 2 - dx2[x, y], 0.0))
        nI = int((Rc[x] & Rc[:, y]).sum())
        out_tau.append(tau)
        out_I.append(nI)
    return np.array(out_tau), np.array(out_I, dtype=float)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(10102026)
    NPTS, NSPR = 2600, 6
    T = 1.0
    Vd = math.pi / 24.0 * T ** 4
    rho = NPTS / Vd
    print(f'засевов {NSPR}, точек по {NPTS}, плотность rho = {rho:.1f}')
    print()
    taus, Is, spr = [], [], []
    for s in range(NSPR):
        P = sprinkle_diamond(NPTS, rs, T)
        Rc, dt, dx2 = rel_matrix(P)
        t_, i_ = sample_pairs(P, Rc, dt, dx2, 4000, rs)
        taus.append(t_)
        Is.append(i_)
        spr.append(np.full(len(t_), s))
    tau = np.concatenate(taus)
    I = np.concatenate(Is)
    sid = np.concatenate(spr)
    print('КОНТРОЛЬ 1: связь |I| с tau — степень и коэффициент')
    m = I >= 20
    a, b = np.polyfit(np.log(tau[m]), np.log(I[m]), 1)
    print(f'  |I| ~ C tau^p:  p = {a:.4f} (ожидание 4), '
          f'C = {math.exp(b):.1f} (ожидание rho*pi/24 = '
          f'{rho*math.pi/24:.1f})')
    print()
    print('ОТНОСИТЕЛЬНАЯ ОШИБКА ОЦЕНКИ tau ПО |I|^{1/4}')
    print(f"{'|I| корзина':>14}{'пар':>7}{'отн. ошибка s':>16}"
          f"{'прогноз 1/(4vN)':>18}{'Phi_max=sqrt2/s':>18}")
    rows = []
    edges = [(20, 40), (40, 80), (80, 160), (160, 320), (320, 640),
             (640, 1300), (1300, 2600)]
    for lo, hi in edges:
        k = (I >= lo) & (I < hi)
        if k.sum() < 120:
            continue
        # оценка: tau_hat = (24 |I| /(pi rho))^{1/4}
        th = (24.0 * I[k] / (math.pi * rho)) ** 0.25
        r = tau[k] / th
        # погрешность по НЕЗАВИСИМЫМ засевам (контроль 2)
        per = [np.std(r[sid[k] == s], ddof=1) / np.mean(r[sid[k] == s])
               for s in range(NSPR) if (sid[k] == s).sum() > 20]
        s_rel = float(np.mean(per))
        s_err = float(np.std(per, ddof=1) / math.sqrt(len(per)))
        nbar = float(np.mean(I[k]))
        pred = 0.25 / math.sqrt(nbar)
        rows.append((nbar, s_rel, s_err, pred))
        print(f'{f"{lo}-{hi}":>14}{int(k.sum()):>7}'
              f'{s_rel:>11.5f}±{s_err:<4.4f}{pred:>18.5f}'
              f'{math.sqrt(2)/s_rel:>18.1f}')
    print()
    R = np.array([(r[0], r[1]) for r in rows])
    pa, pb = np.polyfit(np.log(R[:, 0]), np.log(R[:, 1]), 1)
    print(f'  замеренный закон: s ~ N^{pa:+.4f} '
          f'(прогноз −0.5), коэффициент {math.exp(pb):.4f} '
          f'(прогноз 0.25)')
    print()
    print('КОНТРОЛЬ 4 + ПРЯМОЙ ЗАМЕР: |<exp(i Phi)>| на корзине 320-640')
    k = (I >= 320) & (I < 640)
    th = (24.0 * I[k] / (math.pi * rho)) ** 0.25
    r = tau[k] / th
    r = r / r.mean()
    s_rel = float(r.std(ddof=1))
    print(f'  s = {s_rel:.5f}, пар {int(k.sum())}')
    print(f"      {'Phi':>8}{'|<e^iPhi>| замер':>18}"
          f"{'exp(-Phi^2 s^2/2)':>20}")
    for Phi in (1, 5, 10, 20, 40, 80, 160):
        v = abs(np.exp(1j * Phi * r).mean())
        pred = math.exp(-Phi ** 2 * s_rel ** 2 / 2)
        print(f'      {Phi:>8}{v:>18.5f}{pred:>20.5f}')
    print()
    print('КОНТРОЛЬ 3: искусственно завышенная ошибка (x3) '
          'обязана убить раньше')
    r3 = 1.0 + 3.0 * (r - 1.0)
    for Phi in (10, 40, 160):
        print(f'      Phi={Phi:>4}: обычная {abs(np.exp(1j*Phi*r).mean()):.5f}, '
              f'завышенная {abs(np.exp(1j*Phi*r3).mean()):.5f}')


if __name__ == '__main__':
    main()
