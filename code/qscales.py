#!/usr/bin/env python3
"""ЕСТЬ ЛИ КВАНТОВЫЙ СЛОЙ У АНСАМБЛЯ РАССТАНОВОК ШКАЛ.

6 октября (`quantum-doc.md`): на сприклингах организации фаз НЕТ, и это
ВЫВОДИТСЯ — действие БД есть сумма многих слабо зависимых местных
членов, значит по ЦПТ оно гауссово, а у гауссова действия
характеристическая функция равна exp(i mu/hbar - sigma^2/2 hbar^2)
ТОЧНО, без остатка. Замер подтвердил: худшее отклонение 0.9 сигмы.
Структура появлялась лишь при впуске в ансамбль слоистых порядков — то
есть ввозилась извне.

Но если правило есть ФОРМА (порядок доминирования по k шкалам), а не
сито, ансамбля всех порядков нет, слоистых нет, и ввозить неоткуда.
Тогда сумма идёт не по порядкам, а по РАССТАНОВКАМ ШКАЛ — другой
ансамбль, на который никто не смотрел.

ОЖИДАНИЕ (и потому это проверка там, где предсказание есть точный
нуль): довод ЦПТ применим к любому ансамблю из многих слабо зависимых
членов, значит действие на шкалах тоже обязано быть гауссовым, и
структуры снова не будет. Если так — у формы нет квантового слоя.

НУЛЬ — не формула, а ПОДДЕЛЬНАЯ ВЫБОРКА: n гауссовых действий с тем же
средним и разбросом, так что в нуль входят и затухание, и порог
случайных фаз 1/sqrt(n). Отклонение считается в сигмах разброса нуля.

КОНТРОЛИ:
  1. k=2 доминирование ЕСТЬ двумерный сприклинг (светоконусные
     координаты u=t+x, v=t-x), значит обязано совпасть с засевом d=2;
  2. сприклинговый ансамбль обязан воспроизвести 6 октября (нуль);
  3. при больших hbar обе кривые обязаны идти к единице;
  4. гауссовость проверяется НАПРЯМУЮ: асимметрия и эксцесс.

  python3 qscales.py
"""
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)


def bd_action(Rc):
    """S^(4) = (4/sqrt6)(N - N0 + 9N1 - 16N2 + 8N3)."""
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    N0 = int((Rc & (Iv == 0)).sum())
    N1 = int((Rc & (Iv == 1)).sum())
    N2 = int((Rc & (Iv == 2)).sum())
    N3 = int((Rc & (Iv == 3)).sum())
    return K4 * (n - N0 + 9 * N1 - 16 * N2 + 8 * N3)


def dominance(n, k, rs):
    P = rs.random((n, k))
    return (P[:, None, :] < P[None, :, :]).all(axis=2)


def sprinkle_d(n, d, rs):
    T, pts = 1.0, []
    sd = d - 1
    while len(pts) < n:
        m = int((n - len(pts)) * 6) + 32
        t = rs.random(m) * T
        v = rs.normal(size=(m, sd))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(m) ** (1.0 / sd)) * (T / 2)
        x = v * rad[:, None]
        ok = rad < np.minimum(t, T - t)
        for i in np.nonzero(ok)[0]:
            pts.append(np.concatenate([[t[i]], x[i]]))
    P = np.array(pts[:n])
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt ** 2 > dx2)


def amp(S, hbar):
    return abs(np.exp(1j * S / hbar).mean())


def null_band(mu, sd, n, hbar, rs, B=300):
    vals = [amp(rs.normal(mu, sd, n), hbar) for _ in range(B)]
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def run(name, gen, N, nsamp, rs):
    S = np.array([bd_action(gen()) for _ in range(nsamp)], dtype=float)
    mu, sd = float(S.mean()), float(S.std(ddof=1))
    g1 = float(((S - mu) ** 3).mean() / sd ** 3)
    g2 = float(((S - mu) ** 4).mean() / sd ** 4 - 3.0)
    se1, se2 = math.sqrt(6.0 / nsamp), math.sqrt(24.0 / nsamp)
    print(f'  {name}: <S> = {mu:>12.2f}, sigma = {sd:>9.2f}, '
          f'асимметрия {g1:+.3f}±{se1:.3f}, эксцесс {g2:+.3f}±{se2:.3f}')
    rows = []
    for ratio in (4.0, 1.2, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25):
        hb = ratio * sd
        a = amp(S, hb)
        m, s = null_band(mu, sd, nsamp, hb, rs)
        rows.append((ratio, a, m, s, (a - m) / max(s, 1e-12)))
    return S, mu, sd, rows


def show(rows):
    print(f"      {'hbar/s':>8}{'|amp| замер':>14}{'гауссов нуль':>16}"
          f"{'сигм':>9}")
    worst = 0.0
    for ratio, a, m, s, z in rows:
        print(f'      {ratio:>8.2f}{a:>14.5f}{m:>11.5f} ±{s:<5.4f}'
              f'{z:>9.1f}')
        worst = max(worst, abs(z))
    print(f'      худшее отклонение: {worst:.1f} сигмы')
    return worst


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(10102026)
    N, NS = 300, 2500
    print(f'N = {N}, историй {NS}, действие S^(4)')
    print()
    print('=== КОНТРОЛЬ 2: сприклинги (обязан быть нуль) ===')
    _, _, _, r = run('засев d=4', lambda: sprinkle_d(N, 4, rs), N, NS, rs)
    w_sp4 = show(r)
    print()
    print('=== КОНТРОЛЬ 1: засев d=2 против доминирования k=2 ===')
    _, m1, s1, r1 = run('засев d=2', lambda: sprinkle_d(N, 2, rs), N, NS, rs)
    w1 = show(r1)
    _, m2, s2, r2 = run('доминирование k=2',
                        lambda: dominance(N, 2, rs), N, NS, rs)
    w2 = show(r2)
    print(f'      совпадение: <S> {m1:.1f} против {m2:.1f}, '
          f'sigma {s1:.1f} против {s2:.1f}')
    print()
    print('=== ГЛАВНОЕ: доминирование по k шкалам, k >= 3 ===')
    worst = {}
    for k in (3, 4, 5, 6):
        _, _, _, r = run(f'доминирование k={k}',
                         lambda k=k: dominance(N, k, rs), N, NS, rs)
        worst[k] = show(r)
        print()
    print('=' * 70)
    print('ИТОГ: худшее отклонение от гауссова нуля')
    print(f'  засев d=4 (эталон 6 октября): {w_sp4:.1f} сигмы')
    print(f'  засев d=2: {w1:.1f};  доминирование k=2: {w2:.1f}')
    for k in (3, 4, 5, 6):
        print(f'  доминирование k={k}: {worst[k]:.1f} сигмы')
    big = [k for k in worst if worst[k] > 5]
    print()
    if big:
        print(f'СТРУКТУРА ЕСТЬ при k = {big}: квантовый слой приходит из'
              ' собственной')
        print('свободы конструкции, ввозить слоистые порядки не нужно.')
    else:
        print('СТРУКТУРЫ НЕТ ни при одном k: действие на шкалах тоже')
        print('гауссово, довод ЦПТ подтвердился, и у формы квантового')
        print('слоя НЕТ. Интерференцию придётся ввозить извне.')


if __name__ == '__main__':
    main()
