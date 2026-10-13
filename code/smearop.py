#!/usr/bin/env python3
"""РАЗМАЗАННЫЙ ВОЛНОВОЙ ОПЕРАТОР В 1+1: КОНТРОЛЬ ПЕРЕД ВЕРДИКТОМ.

Зачем. Три замера подряд (waveop5, matter, matter2) показали одно и то
же: РЕЗКИЙ оператор при N <= 1800 шумит сильнее сигнала. Ошибки +-8..47
против целей 2, -2, 4; дельтовость B K всего 1.8-2.9 вместо
бесконечности. Значит сравнивать засев с врагом нечем - «истинный»
объект сам теста не проходит. Размазка у Соркина введена ровно для
этого, и веса я вывел и проверил сегодня же (smear-doc.md).

Вывод нормировок прореживанием (ПЛОТНОСТЬ eps*rho):
    <B' f(x)> = rho [ eps*c_self*f(x) + sum_{y<x} W_m(eps) f(y) ],
    W_m(eps) = eps^2 sum_{j=0..2} w_j C(m,j) eps^j (1-eps)^{m-j}
Собственный член получает eps (выживает один элемент), веса связей -
eps^2 (выживает пара). Это РАЗНЫЕ степени, и именно здесь легко
ошибиться.

ПРАВИЛО ОСТАНОВКИ, объявляю заранее. Размазка проходит, только если
ошибки падают НИЖЕ самих целей (2, -2, 4) при средних, совместимых с
целями. Если нет - материя при доступных N недостижима, и я пишу это
так, а не ищу третью меру. Две меры за сегодня уже умерли от артефактов
(гребень значений, занятость слоёв), третья подгонка была бы
подбором ответа.

КОНТРОЛЬ: при eps = 1 веса обязаны быть (1,-2,1), ноль дальше,
собственный -1/2 - то есть в точности резкий оператор waveop5.
"""
import math

import numpy as np

from waveop import causal, sprinkle_11

W11 = np.array([1.0, -2.0, 1.0])
CSELF = -0.5


def smeared_w(eps, mmax):
    m = np.arange(mmax + 1)
    W = np.zeros(mmax + 1)
    for j in range(3):
        cm = np.array([math.comb(int(mm), j) if mm >= j else 0
                       for mm in m], float)
        W += W11[j] * cm * eps ** j * (1 - eps) ** np.maximum(m - j, 0)
    return eps ** 2 * W


def apply_B(Rc, rho, eps, fs):
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    Warr = smeared_w(eps, int(Iv.max()) + 1)
    M = np.where(Rc.T, Warr[Iv.T], 0.0)
    return [rho * (eps * CSELF * f + M @ f) for f in fs]


def run(n, reps, eps, rs):
    """средние B f в ядре диаманта для f = t^2, x^2, tau^2."""
    acc = [[], [], []]
    for _ in range(reps):
        P = sprinkle_11(n, rs)
        t, x = P[:, 0], P[:, 1]
        Rc = causal(P)
        core = (t > 0.3) & (t < 0.7) & (np.abs(x) < 0.15)
        if core.sum() < 5:
            continue
        outs = apply_B(Rc, n / 0.5, eps, [t ** 2, x ** 2, t ** 2 - x ** 2])
        for a, o in zip(acc, outs):
            a.append(float(o[core].mean()))
    return [(float(np.mean(a)), float(np.std(a) / math.sqrt(len(a))))
            for a in acc]


def main():
    print(__doc__)
    w1 = smeared_w(1.0, 6)
    ok = (abs(w1[0] - 1) < 1e-12 and abs(w1[1] + 2) < 1e-12
          and abs(w1[2] - 1) < 1e-12 and np.abs(w1[3:]).max() < 1e-12)
    print(f'КОНТРОЛЬ: W(eps=1) = {np.round(w1, 6)} '
          f'=> {"ОК" if ok else "СБОЙ"}')

    rs = np.random.default_rng(1618033)
    print('\nсредние B f в ядре, цели: B t^2 = 2, B x^2 = -2, B tau^2 = 4')
    print(f"{'N':>6}{'eps':>7}" + ''.join(f'{h:>19}'
          for h in ('B t^2', 'B x^2', 'B tau^2')))
    for n, reps in ((900, 40), (1800, 18)):
        for eps in (1.0, 0.5, 0.25, 0.1):
            r = run(n, reps, eps, rs)
            print(f'{n:>6}{eps:>7.2f}'
                  + ''.join(f'{m:>12.3f} {e:>6.3f}' for m, e in r),
                  flush=True)
    print('\nЧТЕНИЕ по объявленному правилу: ошибки обязаны уйти НИЖЕ 2')
    print('при средних, совместимых с 2, -2, 4. Иначе материя недостижима')
    print('при этих N, и так и записываем.')


if __name__ == '__main__':
    main()
