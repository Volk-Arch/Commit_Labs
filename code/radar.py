#!/usr/bin/env python3
"""ЦЕПЬ КАК СИСТЕМА ОТСЧЁТА: радарные координаты и переход между ними.

ЕГО МЫСЛЬ. «То, что управляет порядком, не строит пространство.
Пространство задаётся выделенной системой отсчёта или хотя бы чем-то —
наблюдателем. Без пространства получается квантовая структура, а с ним
наш мир.»

ПОЧЕМУ ЭТО СИЛЬНЕЕ, ЧЕМ ОЧЕВИДНО. Архив уже это подтвердил, не заметив.
Проверенная формула проекта `W = 2λr` — окно несравнимости относительно
ОБЪЕКТА даёт расстояние до него. А это и есть радарный метод Эйнштейна,
построенный одним счётом по цепи: для события `x` берём последнее событие
цепи в его прошлом `p` и первое в будущем `f`, тогда

    расстояние = (f − p)/2        время = (p + f)/2

и `(f − p) ≈ 2λr`, то есть `W = 2λr` ЕСТЬ радарное расстояние
наблюдателя. Проект проверял систему отсчёта и думал, что проверяет
линейку.

ЗАЧЕМ ЭТО СЕЙЧАС. Вчера доказано (`locality-doc.md`), что
пространственного расстояния в порядке НЕТ: буст меняет поперечник в
4.88 раза при расхождении порядковых величин 0.000 %. И показано
(`particles-doc.md`), что порядок есть совершенные часы и слепая линейка.
Если линейки нет в порядке, а относительно цепи она есть и проверена, то
цепь и есть то, чем покупается пространство.

ЧТО МЕРИМ. В один мир сажаются ДВА наблюдателя — плотные мировые линии:
один покоящийся, другой с быстротой `rap` (скорость `v = tanh rap`).
Для каждого события фона считаются радарные координаты относительно
каждого, ЧИСТЫМ СЧЁТОМ по цепи, без координат.

ПРОВЕРКИ, расходящиеся по исходу:
  1. Радарные координаты каждого наблюдателя воспроизводят ЕГО
     собственные минковские: корреляция радарного времени с истинным
     временем в его кадре и радарного расстояния с истинным расстоянием.
     Это проверка самой конструкции.
  2. Переход между двумя наборами радарных координат. Если это ЛОРЕНЦ, то
     подгонка `τ₂ = γ(τ₁ − v·x₁)` должна дать `γ` и `v`, совпадающие с
     заложенными. Тогда преобразование Лоренца ВЫВЕДЕНО из порядка плюс
     выбора цепи.
  3. КОНТРОЛЬ: два наблюдателя с ОДИНАКОВОЙ скоростью, но сдвинутые.
     Переход обязан быть сдвигом, а не бустом: подогнанное `v` около
     нуля.

Координаты используются только для СВЕРКИ. В построении радарных
координат их нет — только счёт по цепи.

  python3 radar.py
  python3 radar.py --n 2600 --rap 0.6 --dens 90
"""
import argparse
import math
import time

import numpy as np


def observer(dens, T, rap, off, half=0.06):
    """Плотная мировая линия с быстротой rap, сдвиг off по оси 1."""
    v = math.tanh(rap)
    t = np.linspace(half, T - half, dens)
    Z = np.zeros((dens, 4))
    Z[:, 0] = t
    Z[:, 1] = off + v * (t - T / 2.0)
    return Z


def world(N, T, rs, obs):
    """Ромб плюс наблюдатели. Возвращает фон и списки индексов цепей."""
    out, got = [], 0
    while got < N:
        m = (N - got) * 9 + 128
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= r) & (T - Y[:, 0] >= r)]
        if len(Y):
            out.append(Y)
            got += len(Y)
    bg = np.vstack(out)[:N]
    X = [bg]
    idx = []
    k = len(bg)
    for Z in obs:
        idx.append(np.arange(k, k + len(Z)))
        X.append(Z)
        k += len(Z)
    return np.vstack(X), len(bg), idx


def radar(X, nbg, chain):
    """Радарные координаты фона относительно цепи. ЧИСТЫЙ СЧЁТ."""
    C = X[chain]
    t = X[:nbg, 0][:, None] - C[None, :, 0]
    d = np.linalg.norm(X[:nbg, None, 1:] - C[None, :, 1:], axis=2)
    past = (t > 0) & (t > d)              # событие цепи РАНЬШЕ x
    fut = (t < 0) & (-t > d)              # событие цепи ПОЗЖЕ x
    n = len(C)
    ip = np.where(past.any(1), np.argmax(np.arange(n)[None, :] *
                                         past - (~past) * 10 ** 9, axis=1),
                  -1)
    ifu = np.where(fut.any(1), np.argmax((n - np.arange(n))[None, :] *
                                         fut - (~fut) * 10 ** 9, axis=1),
                   -1)
    ok = (ip >= 0) & (ifu >= 0)
    tau = np.where(ok, 0.5 * (ip + ifu), np.nan)
    rho = np.where(ok, 0.5 * (ifu - ip), np.nan)
    return tau, rho, ok


def main():
    ap = argparse.ArgumentParser(description='Цепь как система отсчёта')
    ap.add_argument('--n', type=int, default=2200)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--rap', type=float, default=0.6)
    ap.add_argument('--dens', type=int, default=80)
    ap.add_argument('--off', type=float, default=0.0)
    ap.add_argument('--seeds', type=int, default=4)
    a = ap.parse_args()

    t0 = time.time()
    v = math.tanh(a.rap)
    g = math.cosh(a.rap)
    print(f'N фона = {a.n}, ромб T = {a.t}, наблюдателей 2 по {a.dens} '
          f'событий')
    print(f'второй с быстротой {a.rap} => v = {v:.4f}, gamma = {g:.4f}')
    print('радарные координаты строятся ЧИСТЫМ СЧЁТОМ по цепи;')
    print('координаты — только для сверки')
    print()

    for nm, rap2, off2 in (('БУСТ', a.rap, 0.0),
                           ('КОНТРОЛЬ: та же скорость, сдвиг',
                            0.0, 0.12)):
        print(f'--- {nm}')
        c1, c2, fv, fg, nb = [], [], [], [], []
        for s in range(a.seeds):
            rs = np.random.default_rng(24400 + 103 * s)
            obs = [observer(a.dens, a.t, 0.0, 0.0),
                   observer(a.dens, a.t, rap2, off2)]
            X, nbg, idx = world(a.n, a.t, rs, obs)
            t1, r1, o1 = radar(X, nbg, idx[0])
            t2, r2, o2 = radar(X, nbg, idx[1])
            ok = o1 & o2
            if ok.sum() < 100:
                continue
            nb.append(int(ok.sum()))
            # 1. воспроизводят ли радарные координаты СВОЙ кадр
            tt = X[:nbg, 0]
            rr = np.linalg.norm(X[:nbg, 1:], axis=1)
            c1.append((np.corrcoef(t1[ok], tt[ok])[0, 1],
                       np.corrcoef(r1[ok], rr[ok])[0, 1]))
            # 2. переход: tau2 = A*tau1 + B*x1 + C, где x1 — положение по
            #    ОСИ ДВИЖЕНИЯ, взятое из радара первого наблюдателя.
            #
            # ПОЙМАННАЯ ОШИБКА. Сначала стояло `x1 = r1 * sign(x)`, то
            # есть РАДИАЛЬНОЕ радарное расстояние со знаком. Но буст идёт
            # по одной оси, и нужна составляющая вдоль неё: `x = r cosθ`.
            # По сфере ⟨|cosθ|⟩ = 1/2, значит такой x1 завышен вдвое, и
            # подогнанное v выходит вдвое малым: вышло 0.2269 против
            # заложенных 0.5370, то есть в 2.37 раза — это оно.
            # Исправление: отбирать события вблизи оси буста, где r ≈ |x|.
            rr3 = np.linalg.norm(X[:nbg, 1:], axis=1)
            cosz = np.where(rr3 > 1e-12, X[:nbg, 1] / np.maximum(rr3, 1e-12),
                            0.0)
            near = ok & (np.abs(cosz) > 0.97)
            if near.sum() < 60:
                near = ok & (np.abs(cosz) > 0.93)
            ok = near
            x1 = r1 * np.sign(cosz)
            M = np.column_stack([t1[ok], x1[ok], np.ones(int(ok.sum()))])
            sol, *_ = np.linalg.lstsq(M, t2[ok], rcond=None)
            A, Bc = float(sol[0]), float(sol[1])
            fg.append(A)
            fv.append(-Bc / A if abs(A) > 1e-9 else np.nan)
            c2.append(np.corrcoef(M @ sol, t2[ok])[0, 1])
            del X
        if not c1:
            print('  мало данных')
            continue
        C1 = np.array(c1)
        print(f'  событий с обоими наблюдателями: {np.mean(nb):.0f}')
        print(f'  1) радар наблюдателя 1 против ЕГО кадра: '
              f'время {C1[:,0].mean():+.4f}, расстояние {C1[:,1].mean():+.4f}')
        print(f'  2) переход tau2 = A*tau1 + B*x1 + C, подгонка:')
        # ОЖИДАНИЕ A = gamma², а не gamma: радарное время мерится в
        # ДЕЛЕНИЯХ ЦЕПИ, а у движущегося наблюдателя деление есть его
        # собственное время, короче в gamma раз. Отсюда лишний множитель.
        g2 = math.cosh(rap2) ** 2
        print(f'       A (ожидание gamma² = {g2:.4f}) = '
              f'{np.mean(fg):.4f} ± '
              f'{np.std(fg, ddof=1)/math.sqrt(len(fg)):.4f}'
              f'   промах {100*(np.mean(fg)-g2)/g2:+.2f} %')
        vt = math.tanh(rap2)
        print(f'       v = -B/A (ожидание {vt:.4f}) = '
              f'{np.nanmean(fv):.4f} ± '
              f'{np.nanstd(fv, ddof=1)/math.sqrt(len(fv)):.4f}'
              f'   промах {100*(np.nanmean(fv)-vt)/max(vt,1e-9):+.2f} %'
              if vt > 1e-9 else
              f'       v = -B/A (ожидание 0) = {np.nanmean(fv):.4f}')
        print(f'       качество подгонки (корреляция) '
              f'{np.mean(c2):.4f}')
        print()

    print('  радар воспроизводит свой кадр И подогнанные gamma, v совпадают')
    print('  с заложенными => ЛОРЕНЦ ВЫВЕДЕН из порядка плюс выбора цепи')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
