#!/usr/bin/env python3
"""ПОСТОЯННЫЕ СВЯЗИ РАЙДАУТА–СОРКИНА: умеет ли поток идти как n^{-3/4}.

`pflow-doc.md` дал числовое требование: эффективная вероятность связи
обязана разбавляться как n^{-(d-1)/d}, а постоянное p (фиксированные
связи перколяционного вида) промахивается на 22 сигмы.

Полная динамика: новый элемент выбирает подмножество S существующих с
вероятностью ∝ t_{|S|}, прошлое есть замыкание ↓S вниз. Формула
проверяется сразу: при t_k = t^k нормировка Σ C(n,k) t^k = (1+t)^n,
то есть ровно бернуллиев выбор с p = t/(1+t) — перколяция. Это и есть
КОНТРОЛЬ правильности реализации.

АНАЛИЗ. Вес w_k = C(n,k) t_k ≈ (n^k/k!) t_k при k << n. Возьмём
t_k = x^k/(k!)^a, тогда ln w_k ≈ k ln(nx) - (a+1) ln k!, и максимум
стоит при

    k* = (nx)^{1/(a+1)},   то есть  E|S| ~ n^{1/(a+1)}

Нужно E|S| ~ n^{1/d}, значит a = d - 1:

    t_k ∝ x^k / (k!)^{d-1}     <- размерность в показателе факториала

ПРЕДРЕГИСТРИРОВАННАЯ ЛЕСТНИЦА (числа записаны ДО прогона):
  a=0: E|S| ~ n^1.00, показатель ширины ~0.13 (перколяция, постоянное p)
  a=1: E|S| ~ n^0.50, показатель ширины ~0.67
  a=2: E|S| ~ n^0.33, показатель ширины между 0.67 и 0.82
  a=3: E|S| ~ n^0.25, показатель ширины ~0.82  <- цель, d=4
Числа ширины взяты из `pflow-doc.md` для alpha = 1 - 1/(a+1).

Для сравнения: засев d=4 даёт показатель ширины 0.8160 ± 0.017.

x подбирается так, чтобы при n=100 среднее |S| было ~2.5 во всех
случаях — иначе сравниваются начальные условия, а не показатели.
"""
import math

import numpy as np
from scipy.special import gammaln, logsumexp

from width3 import width


def logw(n, a, lx):
    k = np.arange(n + 1)
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + k * lx - a * gammaln(k + 1))


def mean_S(n, a, lx):
    lw = logw(n, a, lx)
    p = np.exp(lw - logsumexp(lw))
    return float((np.arange(n + 1) * p).sum())


def tune_x(a, target=2.5, n=100):
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_S(n, a, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def grow(nmax, a, lx, rs, checks):
    Rc = np.zeros((nmax, nmax), bool)
    out, sizes = {}, []
    for n in range(1, nmax):
        lw = logw(n, a, lx)
        p = np.exp(lw - logsumexp(lw))
        k = int(rs.choice(n + 1, p=p))
        sizes.append(k)
        if k:
            S = rs.choice(n, size=k, replace=False)
            sel = np.zeros(n, bool)
            sel[S] = True
            past = sel | Rc[:n, :n][:, sel].any(axis=1)
            Rc[:n, n] = past
        if (n + 1) in checks:
            out[n + 1] = width(Rc[:n + 1, :n + 1])
    return out, sizes


def fit(xs, ys):
    lx_, ly = np.log(np.array(xs, float)), np.log(np.array(ys, float))
    A = np.vstack([lx_, np.ones(len(lx_))]).T
    be, *_ = np.linalg.lstsq(A, ly, rcond=None)
    r = ly - A @ be
    se = math.sqrt(float((r ** 2).sum() / max(len(lx_) - 2, 1))
                   / float(((lx_ - lx_.mean()) ** 2).sum()))
    return float(be[0]), se


def main():
    print(__doc__)
    rs = np.random.default_rng(192837465)
    checks = {125, 250, 500, 1000}
    xs = sorted(checks)
    nmax = 1000

    print('КОНТРОЛЬ: a=0 обязан дать РОВНО перколяцию с постоянным p')
    lx0 = tune_x(0)
    t = math.exp(lx0)
    print(f'  x = {t:.6f}  =>  p = t/(1+t) = {t/(1+t):.6f}')
    for n in (100, 400, 1000):
        print(f'  n={n}: E|S| замер {mean_S(n, 0, lx0):.3f}, '
              f'ожидание n·p = {n*t/(1+t):.3f}')

    print('\nЛЕСТНИЦА: E|S| и показатель ширины')
    print(f"{'a':>3}{'t_k вид':>16}{'E|S| пок.':>12}{'предск.':>9}"
          + ''.join(f'{f"W(n={v})":>10}' for v in xs)
          + f"{'показ. W':>14}")
    pred_w = {0: 0.13, 1: 0.67, 2: 0.74, 3: 0.82}
    for a in (0, 1, 2, 3):
        lx = tune_x(a)
        ns = [125, 250, 500, 1000]
        es = [mean_S(n, a, lx) for n in ns]
        ea, _ = fit(ns, es)
        W = {n: [] for n in checks}
        for _ in range(2):
            out, _sz = grow(nmax, a, lx, rs, checks)
            for n in checks:
                W[n].append(out[n])
        ys = [float(np.mean(W[n])) for n in xs]
        wa, wse = fit(xs, ys)
        print(f'{a:>3}{f"x^k/(k!)^{a}":>16}{ea:>12.4f}'
              f'{1/(a+1):>9.4f}'
              + ''.join(f'{y:>10.1f}' for y in ys)
              + f'{wa:>9.4f}±{wse:<4.3f}', flush=True)

    print('\nСВЕРКА С ПРЕДРЕГИСТРАЦИЕЙ')
    print('  цель d=4: a=3, E|S| ~ n^0.25, показатель ширины ~0.82')
    print('  засев d=4 (замер): 0.8160 ± 0.017')
    print('  перколяция при alpha=3/4 (pflow): 0.8159 ± 0.008')


if __name__ == '__main__':
    main()
