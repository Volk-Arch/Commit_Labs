#!/usr/bin/env python3
"""ТЕЧЁТ ЛИ РАЗМЕРНОСТЬ С МАСШТАБОМ, и что даёт предпочтительное
присоединение.

Два повода. Первый: 10^173 связей на событие в 4d против 560 в 1+1 -
довод в пользу двумерия в основе, и в литературе спектральная
размерность действительно стремится к 2 на малых масштабах. Но днём
(`base2d-doc.md`) ГЛОБАЛЬНО двумерный порядок статистику 4d не даёт
(мимо в 217 раз), значит нужна размерность, ТЕКУЩАЯ с масштабом, - а
этого не проверялось ни разу.

Второй: правила из биологии и химии, работающие на всех уровнях. Точнее
всего для нас ПРЕДПОЧТИТЕЛЬНОЕ ПРИСОЕДИНЕНИЕ: новое цепляется к тому, у
кого уже много связей. Оно даёт степенные законы (а профиль засева и есть
степенной m^{-1/2}), оно локально, и оно КОНТЕКСТНО - то есть выпадает из
теоремы о монотонных правилах (`monotone-doc.md`). Валентность из химии
уже построена - это предел на связи (`cap-doc.md`).

ПРИБОР: доля связанных ВНУТРИ интервала как функция РАЗМЕРА интервала.
Это оценщик размерности (`facecontrol-doc.md`: 0.10 при d=4, 0.50 при
1+1), применённый по масштабам. У засева фиксированной размерности он
обязан быть ПЛОСКИМ - это и есть контроль прибора.
"""
import math

import numpy as np

from sectors import sprinkle_d
from waveop import causal, sprinkle_11
from orthant import scales, thresh_order

N = 1500
BINS = ((3, 6), (6, 12), (12, 25), (25, 50), (50, 120), (120, 10 ** 9))


def spectrum(R, rs, cap=90):
    n = R.shape[0]
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    out = []
    for lo, hi in BINS:
        ii, jj = np.nonzero(R & (Iv >= lo) & (Iv < hi))
        if len(ii) < 20:
            out.append(float('nan'))
            continue
        sel = rs.choice(len(ii), min(cap, len(ii)), replace=False)
        v = []
        for t in sel:
            i, j = ii[t], jj[t]
            m = np.nonzero(R[i] & R[:, j])[0]
            if len(m) >= 3:
                v.append(float(R[np.ix_(m, m)].sum())
                         / (len(m) * (len(m) - 1) / 2))
        out.append(float(np.mean(v)) if v else float('nan'))
    return out


def pref_attach(n, rs, nlink=3, alpha=1.0, a=1.0):
    """предпочтительное присоединение: вероятность ~ (исх.степень + a)^alpha."""
    R = np.zeros((n, n), bool)
    od = np.zeros(n)
    for i in range(1, n):
        w = (od[:i] + a) ** alpha
        w = w / w.sum()
        k = min(nlink, i)
        par = rs.choice(i, size=k, replace=False, p=w)
        anc = np.zeros(n, bool)
        for p in par:
            anc[p] = True
            anc |= R[:, p]
        R[:, i] = anc
        F = anc[:i]
        od[:i] += 0  # исходящие считаем по СВЯЗЯМ ниже, не по отношениям
        od[list(par)] += 1
    return R


def show(lab, sp, extra=''):
    print(f'{lab:>24}' + ''.join(f'{v:>10.3f}' if v == v else f'{"—":>10}'
                                 for v in sp) + f'  {extra}', flush=True)


def main():
    print(__doc__)
    rs = np.random.default_rng(161803)
    print(f'N={N}, доля связанных внутри интервала по размеру интервала')
    print(f'{"ансамбль":>24}' + ''.join(f'{f"{a}–{b}" if b < 1e8 else f">{a}":>10}'
                                        for a, b in BINS))
    print('  КОНТРОЛЬ ПРИБОРА: у засева обязан быть плоским')
    show('засев 1+1 (ждём 0.50)', spectrum(causal(sprinkle_11(N, rs)), rs))
    show('засев 4d (ждём 0.10)', spectrum(sprinkle_d(N, 4, rs), rs))
    show('засев 3d (ждём ~0.23)', spectrum(sprinkle_d(N, 3, rs), rs))
    show('орта k=4', spectrum(thresh_order(scales(N, 4, rs), 4), rs))
    print('  ПРЕДПОЧТИТЕЛЬНОЕ ПРИСОЕДИНЕНИЕ')
    for nl, al in ((3, 1.0), (3, 0.5), (6, 1.0), (6, 0.0)):
        R = pref_attach(N, rs, nlink=nl, alpha=al)
        r = float(R.sum()) / (N * (N - 1) / 2)
        show(f'связей {nl}, alpha={al}', spectrum(R, rs),
             f'доля связ. {r:.3f}')


if __name__ == '__main__':
    main()
