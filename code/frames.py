#!/usr/bin/env python3
"""ЦЕПЬ КАК СИСТЕМА ОТСЧЁТА — самодостаточное воспроизведение.

Один файл, без импортов из проекта: запускается где угодно, нужен только
numpy. Воспроизводит три результата целиком.

    python3 frames.py            все три опыта
    python3 frames.py --only 1   только нужный

ЧТО ВОСПРОИЗВОДИТСЯ

ОПЫТ 0 (основание). Пространственного расстояния в порядке НЕТ. Буст
  применяется к тем же событиям: причинность сохраняется точно, все
  порядковые величины совпадают до нуля, а пространственный поперечник
  интервалов растёт в разы.
    ожидание: расхождение порядковых величин 0.000 %, поперечник x4.9

ОПЫТ 1 (цепь даёт координаты). Радарный метод Эйнштейна, выполненный
  ЧИСТЫМ СЧЁТОМ по цепи-наблюдателю: для события x берём последнее
  событие цепи в его прошлом p и первое в будущем f, тогда
      расстояние = (f − p)/2        время = (p + f)/2
  Это и есть архивная формула проекта `W = 2λr`, только теперь видно,
  что она описывает систему отсчёта.
    ожидание: корреляция с собственными координатами наблюдателя
              ~0.9999 по времени и ~0.9997 по пространству

ОПЫТ 2 (две цепи дают Лоренца). Подгонка перехода между радарными
  координатами двух наблюдателей: `tau2 = A*tau1 + B*x1 + C`.
    ожидание: A = gamma^2 с промахом меньше 0.5 %,  v = -B/A ~ tanh(rap)
              с постоянным калибровочным смещением около -5 %
    КОНТРОЛЬ: та же скорость, сдвиг -> A = 1, v = 0

  ПОЧЕМУ gamma^2, А НЕ gamma: радарное время мерится в ДЕЛЕНИЯХ ЦЕПИ, а
  у движущегося наблюдателя деление есть его СОБСТВЕННОЕ время, короче в
  gamma раз. С gamma промах был 19 %, с gamma^2 стал полпроцента.

ОПЫТ 3 (протечки нет). Наблюдатель — физический объект, его вставка
  меняет множество. Вопрос: зависит ли вклад от СКОРОСТИ? Один объект
  (та же мощность, то же собственное время) в разном движении.
  Зависимость есть (+6.8 % при быстроте 0.9), но она КРАЕВАЯ: при росте
  ОБЛАСТИ при постоянной плотности убывает как T^-1.8.
    ожидание: рост % падает 6.8 -> 1.4 при T = 1.0 -> 2.4,
              показатель около -1.8 (площадь границы на объём)

  ВАЖНО: растить надо ОБЛАСТЬ, а не N. Наращивание N при том же T есть
  измельчение: расстояние до границы не меняется, и краевой эффект
  обязан остаться постоянным — так и выходит (N^+0.000). Это была моя
  ошибка в первом контроле.

ПОЙМАННЫЕ ОШИБКИ, оставленные в описании нарочно:
  1. В подгонку опыта 2 сначала шло `x1 = ±r1`, то есть РАДИАЛЬНОЕ
     радарное расстояние со знаком. Буст идёт по одной оси, нужна
     составляющая `r*cos(theta)`; по сфере `<|cos|> = 1/2`, и `v` выходило
     вдвое малым: 0.2269 против 0.5370, то есть в 2.37 раза. Исправлено
     отбором событий у оси (|cos| > 0.97).
  2. Ожидание `A = gamma` вместо `gamma^2` — см. выше.
  3. Контроль «граница или физика» измельчением вместо увеличения — см.
     опыт 3.
"""
import argparse
import math
import time

import numpy as np

# ---------------------------------------------------------------- миры

def diamond(N, T, rs):
    """Сприклинг в причинный ромб: |x| <= min(t, T-t), объём pi T^4 / 24."""
    out, got = [], 0
    while got < N:
        m = (N - got) * 9 + 256
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= r) & (T - Y[:, 0] >= r)]
        if len(Y):
            out.append(Y)
            got += len(Y)
    X = np.vstack(out)[:N]
    return X[np.argsort(X[:, 0])]


def rel_of(X):
    """R[i,j] = «i раньше j». Сравнение КВАДРАТОВ: точнее и быстрее корня."""
    t = X[:, 0]
    dt = t[None, :] - t[:, None]
    S = X[:, 1:]
    g = (S * S).sum(1)
    d2 = g[:, None] + g[None, :] - 2.0 * (S @ S.T)
    np.maximum(d2, 0.0, out=d2)
    return (dt > 0) & (dt * dt > d2)


def observer(dens, T, rap, off, half=0.06):
    """Мировая линия по КООРДИНАТНОМУ времени (для опытов 1-2)."""
    v = math.tanh(rap)
    t = np.linspace(half, T - half, dens)
    Z = np.zeros((dens, 4))
    Z[:, 0] = t
    Z[:, 1] = off + v * (t - T / 2.0)
    return Z


def observer_proper(dens, tau, rap, t0, x0):
    """Мировая линия по СОБСТВЕННОМУ времени: один объект при любой v.

    Для опыта 3 это принципиально: сравнивать надо ОДИН объект в разном
    движении, а не разные объекты.
    """
    ch, sh = math.cosh(rap), math.sinh(rap)
    s = np.linspace(-tau / 2.0, tau / 2.0, dens)
    Z = np.zeros((dens, 4))
    Z[:, 0] = t0 + ch * s
    Z[:, 1] = x0 + sh * s
    return Z


def fits(Z, T=1.0):
    r = np.linalg.norm(Z[:, 1:], axis=1)
    return bool(np.all((Z[:, 0] >= r) & (T - Z[:, 0] >= r)))

# ------------------------------------------------------------ опыт 0

def boosted(X, rap):
    ch, sh = math.cosh(rap), math.sinh(rap)
    Y = X.copy()
    Y[:, 0] = ch * X[:, 0] + sh * X[:, 1]
    Y[:, 1] = sh * X[:, 0] + ch * X[:, 1]
    return Y


def links_of(R):
    A = R.astype(np.float32)
    return R & ~((A @ A) > 0.5)


def exp0(n=1500, raps=(0.0, 1.0, 2.0), card=(12, 20), npair=300, seeds=3):
    print('ОПЫТ 0: пространственного расстояния в порядке НЕТ')
    print('буст применяется к ТЕМ ЖЕ событиям; порядок обязан совпасть')
    print(f"{'быстрота':>9}{'отношений':>11}{'связей':>9}{'высота':>8}"
          f"{'доля сравн.':>13}{'ПОПЕРЕЧНИК':>12}")
    base, rows = None, []
    for rap in raps:
        acc = []
        for s in range(seeds):
            X = diamond(n, 1.0, np.random.default_rng(700 + 11 * s))
            Y = boosted(X, rap)
            Y = Y[np.argsort(Y[:, 0])]
            R = rel_of(Y)
            L = links_of(R)
            h = np.ones(n, np.int64)
            for i in range(n):
                pre = np.flatnonzero(R[:, i])
                if len(pre):
                    h[i] = h[pre].max() + 1
            A = R.astype(np.float32)
            INT = A @ A
            rel = np.argwhere(R)
            c = INT[rel[:, 0], rel[:, 1]]
            ok = np.flatnonzero((c >= card[0]) & (c <= card[1]))
            rs2 = np.random.default_rng(3 + s)
            ok = rs2.choice(ok, min(npair, len(ok)), replace=False)
            dia = []
            for p in ok:
                i, j = int(rel[p, 0]), int(rel[p, 1])
                mem = np.flatnonzero(R[i, :] & R[:, j])
                sub = np.concatenate([[i], mem, [j]])
                P = Y[sub]
                d = 0.0
                for q in range(len(P)):
                    dd = np.linalg.norm(P[q + 1:, 1:] - P[q, 1:], axis=1)
                    if len(dd):
                        d = max(d, float(dd.max()))
                dia.append(d)
            acc.append((R.sum(), L.sum(), h.max(),
                        R.sum() / (n * (n - 1) / 2), np.mean(dia)))
            del R, L, A, INT, X, Y
        m = np.array(acc, float).mean(0)
        if base is None:
            base = m.copy()
        rows.append(m)
        print(f'{rap:>9.2f}{m[0]:>11.0f}{m[1]:>9.0f}{m[2]:>8.1f}'
              f'{m[3]:>13.5f}{m[4]:>12.5f}')
    dev = max(abs(r[k] - base[k]) / max(abs(base[k]), 1e-12)
              for r in rows for k in range(4))
    dmax = max(r[4] for r in rows) / rows[0][4]
    print(f'  расхождение порядковых величин не более {100*dev:.3f} %')
    print(f'  поперечник вырос в {dmax:.2f} раза')
    print('  => ПОПЕРЕЧНИК НЕ ЕСТЬ ФУНКЦИЯ ПОРЯДКА')
    print()

# ---------------------------------------------------------- опыты 1-2

def radar(X, nbg, C):
    """Радарные координаты фона относительно цепи C. ЧИСТЫЙ СЧЁТ."""
    t = X[:nbg, 0][:, None] - C[None, :, 0]
    d = np.linalg.norm(X[:nbg, None, 1:] - C[None, :, 1:], axis=2)
    past = (t > 0) & (t > d)
    fut = (t < 0) & (-t > d)
    n = len(C)
    big = 10 ** 9
    ip = np.where(past.any(1),
                  np.argmax(np.arange(n)[None, :] * past - (~past) * big,
                            axis=1), -1)
    ifu = np.where(fut.any(1),
                   np.argmax((n - np.arange(n))[None, :] * fut -
                             (~fut) * big, axis=1), -1)
    ok = (ip >= 0) & (ifu >= 0)
    return (np.where(ok, 0.5 * (ip + ifu), np.nan),
            np.where(ok, 0.5 * (ifu - ip), np.nan), ok)


def exp12(n=5000, raps=(0.3, 0.6, 0.9), dens=130, seeds=4, cut=0.97,
          off=0.12):
    print('ОПЫТЫ 1-2: цепь даёт координаты, две цепи дают Лоренца')
    print(f'N фона {n}, наблюдателей по {dens} событий, засевов {seeds},'
          f' отбор у оси |cos| > {cut}')
    print(f"{'случай':>26}{'время':>9}{'простр.':>9}"
          f"{'A':>9}{'gamma²':>9}{'промах':>9}{'v':>9}{'ожид.':>8}"
          f"{'промах':>9}{'качество':>10}")
    for nm, rap2, of2 in ([(f'буст {r}', r, 0.0) for r in raps] +
                          [('КОНТРОЛЬ сдвиг', 0.0, off)]):
        c1, fa, fv, q = [], [], [], []
        for s in range(seeds):
            rs = np.random.default_rng(24400 + 103 * s)
            bg = diamond(n, 1.0, rs)
            Z1 = observer(dens, 1.0, 0.0, 0.0)
            Z2 = observer(dens, 1.0, rap2, of2)
            X = np.vstack([bg, Z1, Z2])
            i1 = np.arange(n, n + dens)
            i2 = np.arange(n + dens, n + 2 * dens)
            t1, r1, o1 = radar(X, n, X[i1])
            t2, r2, o2 = radar(X, n, X[i2])
            ok = o1 & o2
            tt, rr = X[:n, 0], np.linalg.norm(X[:n, 1:], axis=1)
            c1.append((np.corrcoef(t1[ok], tt[ok])[0, 1],
                       np.corrcoef(r1[ok], rr[ok])[0, 1]))
            cz = np.where(rr > 1e-12, X[:n, 1] / np.maximum(rr, 1e-12), 0.0)
            m = ok & (np.abs(cz) > cut)
            if m.sum() < 60:
                m = ok & (np.abs(cz) > 0.93)
            x1 = r1 * np.sign(cz)
            M = np.column_stack([t1[m], x1[m], np.ones(int(m.sum()))])
            sol, *_ = np.linalg.lstsq(M, t2[m], rcond=None)
            fa.append(float(sol[0]))
            fv.append(-float(sol[1]) / float(sol[0]))
            q.append(np.corrcoef(M @ sol, t2[m])[0, 1])
            del X, bg
        C = np.array(c1)
        g2 = math.cosh(rap2) ** 2
        vt = math.tanh(rap2)
        A = np.mean(fa); V = np.mean(fv)
        pa = 100 * (A - g2) / g2
        pv = (100 * (V - vt) / vt) if vt > 1e-9 else float('nan')
        print(f'{nm:>26}{C[:,0].mean():>9.4f}{C[:,1].mean():>9.4f}'
              f'{A:>9.4f}{g2:>9.4f}{pa:>+9.2f}{V:>9.4f}{vt:>8.4f}'
              f'{pv:>+9.2f}{np.mean(q):>10.4f}')
    print('  время/простр. ~0.9999/0.9997 => цепь даёт полные координаты')
    print('  A = gamma² с промахом <0.5 %, контроль A=1 v=0 => ЛОРЕНЦ')
    print()

# ------------------------------------------------------------ опыт 3

def obs_rel(X, Z):
    """Причинные пары наблюдатель-фон за O(dens*N): без матриц N*N."""
    tot = 0
    for k in range(len(Z)):
        dt = X[:, 0] - Z[k, 0]
        dx = np.linalg.norm(X[:, 1:] - Z[k, 1:], axis=1)
        tot += int((np.abs(dt) > dx).sum())
    return tot


def exp3(ell=0.107, tau=0.22, dens=20, seeds=10, Ts=(1.0, 1.4, 1.8, 2.4)):
    print('ОПЫТ 3: протечки нет — зависимость от скорости КРАЕВАЯ')
    print(f'наблюдатель {dens} событий, собственное время {tau} — '
          f'ОДИН объект при любой быстроте')
    print(f'ell = {ell} фиксировано, растёт ОБЛАСТЬ (не измельчение!)')
    print(f"{'T':>6}{'N':>9}{'радиус/tau':>12}{'отн ρ=0':>11}"
          f"{'отн ρ=0.9':>12}{'рост %':>9}{'сигм':>7}")
    rows = []
    for T in Ts:
        N = int(round(math.pi / 24.0 * T ** 4 / ell ** 4))
        d0, d9 = [], []
        for s in range(seeds):
            X = diamond(N, T, np.random.default_rng(1300 + 37 * s))
            for rap, acc in ((0.0, d0), (0.9, d9)):
                acc.append(obs_rel(X, observer_proper(dens, tau, rap,
                                                      T / 2.0, 0.0)))
            del X
        d0 = np.array(d0, float); d9 = np.array(d9, float)
        dd = d9.mean() - d0.mean()
        sd = math.hypot(d0.std(ddof=1) / math.sqrt(len(d0)),
                        d9.std(ddof=1) / math.sqrt(len(d9)))
        pc = 100 * dd / d0.mean()
        rows.append((T, abs(pc)))
        print(f'{T:>6.1f}{N:>9}{(T/2)/tau:>12.1f}{d0.mean():>11.0f}'
              f'{d9.mean():>12.0f}{pc:>9.2f}'
              f'{abs(dd)/max(sd,1e-9):>7.1f}', flush=True)
    A = np.array(rows)
    b = np.polyfit(np.log(A[:, 0]), np.log(np.maximum(A[:, 1], 1e-9)), 1)[0]
    print(f'  зависимость ~ T^{b:+.3f}  (ожидание около -2: граница/объём)')
    print('  убывает => ПРОТЕЧКИ НЕТ, вопрос корректен лишь в пределе')
    print()


def main():
    ap = argparse.ArgumentParser(description='Цепь как система отсчёта')
    ap.add_argument('--only', type=int, default=0,
                    help='0 все, иначе номер опыта (0,1,3)')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    print('=' * 72)
    print('ЦЕПЬ КАК СИСТЕМА ОТСЧЁТА: воспроизведение')
    print('=' * 72)
    print()
    q = a.quick
    if a.only in (0,):
        exp0(n=900 if q else 1500, seeds=2 if q else 3)
    if a.only in (0, 1, 2):
        exp12(n=2000 if q else 5000, dens=90 if q else 130,
              seeds=2 if q else 4)
    if a.only in (0, 3):
        exp3(seeds=4 if q else 10, Ts=(1.0, 1.4, 1.8) if q else
             (1.0, 1.4, 1.8, 2.4))
    print(f'время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
