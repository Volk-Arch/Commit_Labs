#!/usr/bin/env python3
"""ДВА ЧЛЕНА ПОРОЗНЬ: B из семейства с R = 0, затем A.

ОТКУДА. `radarcell-doc.md`: кривизна измерена чистым счётом и сошлась с
формулой Гиббонса–Солодухина (контроль плоского 1.1 сигмы, согласие 1.5),
но `A` вышло до 6 %, а `B` лишь до 26 % — он держался на коротком рычаге
разницы скоростей (`R_uu` менялся от −3c до −4.9c).

Давить скоростями дорого: при `v = 0.85` ромб вытягивается на
`Δη = 1.9τ`, и ящик нужен втрое шире. Но у задачи есть лучший рычаг, и он
в устройстве самой метрики.

В конформно-плоской метрике `R = 6a''/a³`. Значит при ЛИНЕЙНОМ
`a = 1 + h(η−η₀)` выходит `a'' = 0` и `R = 0` ТОЖДЕСТВЕННО, везде, — а
тензор Риччи при этом не нуль (это FRW с излучением: след нулевой, сам
тензор нет). Проверено символьно:

    линейное:   R = 0,  R_ηη = 3h²/a²,  R_xx = h²/a²
                R_uu(v) = h²(3+v²) / (a⁴(1−v²))
    квадратич:  R = 6c, R_ηη = −3c, R_xx = c  (в центре)
                R_uu(v) = c(v²−3) / (1−v²)

Тогда

    линейное:   dev = τ²·B·R_uu            =>  B НАПРЯМУЮ, без A
    квадратич:  dev = τ²(A·R + B·R_uu)     =>  A при известном B

Это и точнее, и дешевле. Побочная выгода: при `R = 0` везде ограничитель
профиля не нужен вовсе, значит третья пойманная ошибка (ромб залезал в
плоскую зону) повториться не может.

Цена: `R_uu ∝ h²`, то есть сигнал квадратичен по параметру и слабее.
Компенсируется числом засевов и скоростью (при `v = 0.7` множитель
`(3+v²)/(1−v²)` даёт 6.84 против 3 при нуле — рычаг 2.3 раза).

КОНТРОЛИ, поставленные до замера:
  1. нулевой параметр (h = 0 или c = 0) обязан дать dev = 0 при ЛЮБОЙ
     скорости — это и нулевая точка, и проверка нормировки объёма;
  2. ромб обязан целиком лежать в засеянной области (проверяется);
  3. параметр разложения `τ²·|кривизна|` печатается — формула первого
     порядка требует, чтобы он был мал, и это надо видеть, а не
     предполагать;
  4. ПРОВЕРКА ВЫСШИХ ПОРЯДКОВ: отношение замер/формула при разных
     значениях параметра. Если недостача уходит с уменьшением кривизны —
     остаток объяснён высшими порядками, а не ошибкой;
  5. `B` из линейного семейства сравнивается с `B`, вытащенным из
     квадратичного вместе с `A`: два независимых пути к одному числу.

  python3 twoterm.py
"""
import math
import time

import numpy as np

LELL = 22.0
TELL = 20.0
ETA0 = TELL / 2.0


def a_of(eta, fam, par):
    if fam == 'lin':
        return 1.0 + par * (eta - ETA0)
    return 1.0 + par * (eta - ETA0) ** 2 / 2.0


def curv(eta, fam, par, pmom):
    """R и R_uu в точке eta для ГЕОДЕЗИЧЕСКОЙ с импульсом pmom.

    ПОЙМАННАЯ ОШИБКА. Сначала подставлялась номинальная скорость `v` для
    всех ромбов. Но у геодезической сохраняется не скорость, а импульс
    `p = a²·dx/dτ`, и координатная скорость `v(eta) = p/sqrt(a²+p²)`
    меняется вдоль пути: при линейном a = 1+0.06(eta-eta0) и v(eta0)=0.7
    на смещении 4 она падает до 0.62. Для смещённых ромбов это давало не
    ту кривизну, и «починка» ухудшила ответ (B с 2 % до 32 %).

    Правильно, через сохраняющийся p:
        u^mu = (sqrt(a²+p²)/a², p/a², 0, 0),   u·u = -1 тождественно
        R_uu = [ R_ee·(a²+p²) + R_xx·p² ] / a⁴
    При a = 1 это даёт [R_ee + v²R_xx]/(1-v²) — прежнее выражение.
    """
    a = a_of(eta, fam, par)
    if fam == 'lin':
        R = 0.0
        Ree = 3.0 * par ** 2 / a ** 2
        Rxx = par ** 2 / a ** 2
    else:
        R = 6.0 * par / a ** 3
        Ree = 6.0 * par * (par * (eta - ETA0) ** 2 - 2) / (
            (par * (eta - ETA0) ** 2 + 2) ** 2) * 1.0
        Rxx = 2.0 * par * (3 * par * (eta - ETA0) ** 2 + 2) / (
            (par * (eta - ETA0) ** 2 + 2) ** 2) * 1.0
    Ruu = (Ree * (a * a + pmom * pmom) + Rxx * pmom * pmom) / a ** 4
    return R, Ruu


def sprinkle(fam, par, rs):
    e = np.linspace(0.0, TELL, 20000)
    w = a_of(e, fam, par) ** 4
    if (a_of(e, fam, par) <= 0).any():
        raise ValueError('a ушло в нуль: параметр слишком велик')
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                           * np.diff(e))])
    need = int(round(LELL ** 3 * cdf[-1]))
    u = rs.uniform(0, cdf[-1], need)
    X = np.empty((need, 4))
    X[:, 0] = np.interp(u, cdf, e)
    X[:, 1:] = rs.uniform(0, LELL, (need, 3))
    return X[np.argsort(X[:, 0])], need / (LELL ** 3 * cdf[-1])


def chain(fam, par, v, step, n, xoff):
    """Цепь вдоль НАСТОЯЩЕЙ ГЕОДЕЗИЧЕСКОЙ, узлы через равное собственное время.

    ПОЙМАННАЯ ОШИБКА. Раньше цепь шла с постоянной КООРДИНАТНОЙ
    скоростью. Но в конформно-плоской метрике такая линия НЕ
    геодезическая при v ≠ 0, а формула Гиббонса–Солодухина требует
    геодезического собственного времени между остриями. Значит `tau`
    брался не тот.

    Правильно: метрика не зависит от x, значит сохраняется `p = a²·dx/dτ`.
    Из нормировки `a²(η'² − x'²) = 1` выходит

        dx/deta = p / sqrt(a² + p²),     dtau/deta = a² / sqrt(a² + p²)

    и при `a = 1` (в центре) скорость равна `p/sqrt(1+p²)`, то есть
    `p = v/sqrt(1-v²)` — обычная связь. Узлы ставятся через равное
    собственное время ПО ЭТОЙ линии.
    """
    pmom = v / math.sqrt(1 - v * v) if v > 0 else 0.0
    e = np.linspace(0.0, TELL, 20000)
    a = a_of(e, fam, par)
    den = np.sqrt(a * a + pmom * pmom)
    dxde = pmom / den
    dtde = a * a / den
    xs = np.concatenate([[0.0], np.cumsum(0.5 * (dxde[1:] + dxde[:-1])
                                          * np.diff(e))])
    pt = np.concatenate([[0.0], np.cumsum(0.5 * (dtde[1:] + dtde[:-1])
                                          * np.diff(e))])
    tot = pt[-1]
    t0 = 0.5 * (tot - (n - 1) * step)
    if t0 < 0:
        raise ValueError('цепь не влезает')
    want = t0 + step * np.arange(n)
    etas = np.interp(want, pt, e)
    # x отсчитывается так, чтобы геодезическая проходила через центр
    x_at_eta0 = float(np.interp(ETA0, e, xs))
    C = np.zeros((n, 4))
    C[:, 0] = etas
    C[:, 1] = LELL / 2 + xoff + (np.interp(etas, e, xs) - x_at_eta0)
    C[:, 2] = LELL / 2
    C[:, 3] = LELL / 2
    return C


def count(X, p, f):
    t = X[:, 0]
    i0 = np.searchsorted(t, p[0], 'right')
    i1 = np.searchsorted(t, f[0], 'left')
    if i1 <= i0:
        return 0
    S = X[i0:i1, 1:]
    tt = t[i0:i1]
    d1 = S - p[1:]
    d2 = f[1:] - S
    return int((((tt - p[0]) ** 2 > (d1 * d1).sum(1))
                & ((f[0] - tt) ** 2 > (d2 * d2).sum(1))).sum())


def measure(fam, pars, vs, gaps, step, nchain, xoffs, seeds, t0):
    acc = {}
    for par in pars:
        for sd in range(seeds):
            X, rho = sprinkle(fam, par, np.random.default_rng(9100 + 53 * sd))
            for v in vs:
                pmom = v / math.sqrt(1 - v * v) if v > 0 else 0.0
                for xoff in xoffs:
                    C = chain(fam, par, v, step, nchain, xoff)
                    for g in gaps:
                        tau = g * step
                        for i in range(nchain - g):
                            mid = 0.5 * (C[i, 0] + C[i + g, 0])
                            if abs(mid - ETA0) > 4.0:
                                continue
                            de = C[i + g, 0] - C[i, 0]
                            xlo = min(C[i, 1], C[i + g, 1]) - de / 2
                            xhi = max(C[i, 1], C[i + g, 1]) + de / 2
                            if xlo < 0.5 or xhi > LELL - 0.5:
                                continue
                            if C[i, 0] < 0.5 or C[i + g, 0] > TELL - 0.5:
                                continue
                            n_in = count(X, C[i], C[i + g])
                            V = n_in / rho
                            dev = V / (math.pi / 24 * tau ** 4) - 1.0
                            # ПОЙМАННАЯ ОШИБКА: кривизна бралась в eta0,
                            # а ромбы стоят по всему окну, где R = 6c/a³
                            # отличается на четверть. Берём кривизну в
                            # центре КАЖДОГО ромба.
                            Rc, Ruc = curv(mid, fam, par, pmom)
                            acc.setdefault((par, v, g), []).append(
                                (dev, tau, Rc, Ruc))
            del X
        print(f'    {fam} par={par}: готово ({time.time()-t0:.0f} с)',
              flush=True)
    return acc


def report(fam, acc, pars, vs, gaps, step, Bfix=None):
    print(f"{'пар':>8}{'v':>7}{'tau':>7}{'R':>9}{'R_uu':>10}{'ромбов':>8}"
          f"{'dev':>10}{'ст.ош':>9}{'τ²|крив|':>10}")
    rows = []
    for par in pars:
        for v in vs:
            for g in gaps:
                k = (par, v, g)
                if k not in acc or len(acc[k]) < 10:
                    continue
                arr = np.array(acc[k])
                a = arr[:, 0]
                dev, se = a.mean(), a.std(ddof=1) / math.sqrt(len(a))
                tau = g * step
                R = arr[:, 2].mean()        # средняя по ромбам клетки
                Ruu = arr[:, 3].mean()
                rows.append((par, v, g, tau, R, Ruu, dev, se, len(a)))
                print(f'{par:>8.3f}{v:>7.2f}{tau:>7.2f}{R:>9.5f}{Ruu:>10.5f}'
                      f'{len(a):>8}{dev:>+10.5f}{se:>9.5f}'
                      f'{tau**2*max(abs(R),abs(Ruu)):>10.2f}')
    return rows


def main():
    t0 = time.time()
    step, nchain = 1.4, 9
    gaps = (3, 4)
    vs = (0.0, 0.45, 0.70)
    xoffs = (-4.5, -1.5, 1.5, 4.5)
    seeds = 48

    print('ЛИНЕЙНОЕ СЕМЕЙСТВО: R = 0 тождественно, значит dev = τ²·B·R_uu')
    print('и B меряется НАПРЯМУЮ, без примеси от A.')
    print()
    hs = (0.0, 0.04, 0.06)
    accL = measure('lin', hs, vs, gaps, step, nchain, xoffs, seeds, t0)
    print()
    rowsL = report('lin', accL, hs, vs, gaps, step)

    use = [r for r in rowsL if r[0] != 0.0]
    M = np.array([[r[3] ** 2 * r[5]] for r in use])
    y = np.array([r[6] for r in use])
    w = 1.0 / np.array([r[7] for r in use])
    B, *_ = np.linalg.lstsq(M * w[:, None], y * w, rcond=None)
    B = float(B[0])
    # ошибка складным ножом по строкам
    jk = []
    for i in range(len(use)):
        k = [j for j in range(len(use)) if j != i]
        b, *_ = np.linalg.lstsq(M[k] * w[k][:, None], y[k] * w[k], rcond=None)
        jk.append(float(b[0]))
    jk = np.array(jk)
    seB = math.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean()) ** 2).sum())
    print()
    print(f'  B из линейного семейства: {B:+.5f} +- {seB:.5f}')
    print(f'  опубликовано 1/30 =       {1/30:+.5f}   '
          f'({abs(B-1/30)/max(seB,1e-12):.1f} сигм, '
          f'промах {abs(B-1/30)/(1/30)*100:.0f} %)')
    z = [r for r in rowsL if r[0] == 0.0]
    if z:
        print(f'  КОНТРОЛЬ h = 0: худшее {max(abs(r[6]/r[7]) for r in z):.1f} '
              f'сигм')

    print()
    print('=' * 72)
    print('КВАДРАТИЧНОЕ СЕМЕЙСТВО при известном B: вытаскиваем A')
    print()
    cs = (0.0, 0.004, 0.012)
    accQ = measure('quad', cs, vs, gaps, step, nchain, xoffs, seeds, t0)
    print()
    rowsQ = report('quad', accQ, cs, vs, gaps, step)
    useQ = [r for r in rowsQ if r[0] != 0.0]
    # A из остатка после вычитания известного члена B
    num = np.array([r[6] - r[3] ** 2 * B * r[5] for r in useQ])
    den = np.array([r[3] ** 2 * r[4] for r in useQ])
    wq = 1.0 / np.array([r[7] for r in useQ])
    A = float(np.sum(num * den * wq ** 2) / np.sum(den * den * wq ** 2))
    jk = []
    for i in range(len(useQ)):
        k = [j for j in range(len(useQ)) if j != i]
        jk.append(float(np.sum(num[k] * den[k] * wq[k] ** 2)
                        / np.sum(den[k] * den[k] * wq[k] ** 2)))
    jk = np.array(jk)
    seA = math.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean()) ** 2).sum())
    print()
    print(f'  A из квадратичного (при B из линейного): {A:+.5f} +- {seA:.5f}')
    print(f'  опубликовано -1/180 =                    {-1/180:+.5f}   '
          f'({abs(A+1/180)/max(seA,1e-12):.1f} сигм, '
          f'промах {abs(A+1/180)/(1/180)*100:.0f} %)')
    zq = [r for r in rowsQ if r[0] == 0.0]
    if zq:
        print(f'  КОНТРОЛЬ c = 0: худшее '
              f'{max(abs(r[6]/r[7]) for r in zq):.1f} сигм')

    print()
    print('ПРОВЕРКА ВЫСШИХ ПОРЯДКОВ: отношение замер/формула против')
    print('параметра разложения. Если недостача уходит с уменьшением')
    print('кривизны — остаток объяснён высшими порядками.')
    print(f"{'семейство':>11}{'пар':>8}{'v':>7}{'τ²|крив|':>10}"
          f"{'замер':>10}{'формула':>10}{'отнош':>8}{'сигм':>7}")
    for nm, rows in (('линейное', rowsL), ('квадратич', rowsQ)):
        for r in rows:
            if r[0] == 0.0:
                continue
            pred = r[3] ** 2 * (-1 / 180.0 * r[4] + 1 / 30.0 * r[5])
            if abs(pred) < 1e-9:
                continue
            print(f'{nm:>11}{r[0]:>8.3f}{r[1]:>7.2f}'
                  f'{r[3]**2*max(abs(r[4]),abs(r[5])):>10.2f}'
                  f'{r[6]:>+10.5f}{pred:>+10.5f}{r[6]/pred:>8.3f}'
                  f'{abs(r[6]-pred)/r[7]:>7.1f}')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
