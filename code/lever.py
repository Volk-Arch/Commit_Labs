#!/usr/bin/env python3
"""ВИЛКА ДО ПРАВИЛА: достижима ли вообще B2?

ПРАВИЛО, по которому это делается (из `ARTICLE.md`): «вилку мерить до
правила — прежде чем настраивать, спросить, способно ли семейство в
принципе». Прошлая попытка B2 провалила свой контроль: базис `φ_j`
почти коллинеарен, ошибка `c₀` раздувалась с ±0.0019 до ±0.0552, а
половина-против-половины дала 1.8 % при сигнале 0.17 %. Поэтому сначала
считается ЗАКОН ТОЧНОСТИ, и только потом — мерить или не мерить.

ПРИНЦИПИАЛЬНАЯ ПОЧИНКА. Вместо подгонки по усечённому базису строится
линейный функционал от `{F_k}` с МИНИМАЛЬНОЙ ДИСПЕРСИЕЙ при точных
условиях несмещённости:

    оценка = Σ_k w_k F_k,
    Σ_k w_k φ_0(k) = 1,   Σ_k w_k φ_j(k) = 0  (j = 1..J),
    минимизируем  wᵀ Σ w,   Σ = Cov(F_k, F_k')

Решение: `w = Σ⁻¹A(AᵀΣ⁻¹A)⁻¹e₀`. Это наилучшее, что вообще можно
извлечь из `F_k` при данном числе удерживаемых членов, поэтому если и
оно не дотягивает — B2 этим прибором недостижима, и дальше мерить
бессмысленно.

ЧТО СРАВНИВАЕТСЯ. Предсказание (`predict.py`):

    dA/A = −(1/60)·τ²R_uu,

то есть при `τ²R_uu = 0.1` сигнал равен **0.17 %**. Нужно 3 σ, значит
`σ(A)/A ≤ 0.056 %` на всю выборку. Отсюда и число ромбов.

ΣCov берётся ЗАМЕРОМ на плоских ромбах, а не моделью: `F_k` сильно
скоррелированы между собой (элемент с `k` событиями в будущем влияет на
соседние `k`), и пуассоновское приближение тут завысило бы точность.

  python3 lever.py
"""
import math
import time

import numpy as np

from sprinkle import sprinkle_d

KMAX = 26


def fvec(A):
    """F_k: число элементов ровно с k событиями в будущем, k = 0..KMAX-1."""
    nf = A.sum(axis=1)
    return np.bincount(nf, minlength=KMAX)[:KMAX].astype(float)


def sample(n, reps, root):
    out = []
    for s in range(reps):
        A = sprinkle_d(n, 4, np.random.default_rng(root + 17 * s))
        out.append(fvec(A))
        del A
    return np.array(out, float)


def phis(J):
    k = np.arange(KMAX, dtype=float)
    cols = []
    for j in range(J + 1):
        cols.append(np.exp([math.lgamma(kk + 0.5 + j / 2.0)
                            - math.lgamma(kk + 1.0) for kk in k]))
    return np.column_stack(cols)


def best(Sig, A):
    """Минимальная дисперсия при Aᵀw = e_0."""
    Si = np.linalg.pinv(Sig)
    M = A.T @ Si @ A
    e0 = np.zeros(A.shape[1]); e0[0] = 1.0
    Mi = np.linalg.pinv(M)
    w = Si @ A @ (Mi @ e0)
    return w, float(w @ Sig @ w)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 100)
    print('ЗАМЕР ковариации F_k на плоских ромбах и достижимая точность')
    print(f"{'C₁':>7}{'реализ':>8}{'J':>4}{'c₀':>12}{'σ(c₀) 1 ромб':>15}"
          f"{'отн.':>9}{'число обусл.':>15}{'ромбов для 3σ':>16}")
    TARGET = 0.0017 / 3.0          # нужная относительная точность площади
    for n in (400, 800, 1600):
        F = sample(n, 160, 3000 + n)
        mu = F.mean(axis=0)
        Sig = np.cov(F.T)
        for J in (0, 1, 2, 3, 4):
            Aj = phis(J)
            w, var = best(Sig, Aj)
            c0 = float(w @ mu)
            sd = math.sqrt(max(var, 0.0))
            rel = sd / abs(c0) if c0 != 0 else float('inf')
            cond = np.linalg.cond(Aj.T @ Aj)
            nd = (rel / TARGET) ** 2
            print(f'{n:>7}{len(F):>8}{J:>4}{c0:>12.3f}{sd:>15.4f}'
                  f'{100*rel:>8.2f}%{cond:>15.2e}{nd:>16.3g}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 100)
    print('ПРОВЕРКА САМОЙ ОЦЕНКИ: идёт ли c₀ как C₁^(1/2)? (площадь обязана)')
    print(f"{'J':>4}" + ''.join(f'{f"C₁={n}":>14}' for n in (400, 800, 1600))
          + f"{'показатель':>14}")
    MU, SG = {}, {}
    for n in (400, 800, 1600):
        F = sample(n, 160, 3000 + n)
        MU[n], SG[n] = F.mean(axis=0), np.cov(F.T)
    for J in (0, 1, 2, 3, 4):
        Aj = phis(J)
        vals, errs = [], []
        for n in (400, 800, 1600):
            w, var = best(SG[n], Aj)
            vals.append(float(w @ MU[n]))
            errs.append(math.sqrt(max(var, 0.0) / 160))
        x = np.log([400, 800, 1600]); y = np.log(np.abs(vals))
        k = np.polyfit(x, y, 1)[0]
        print(f'{J:>4}' + ''.join(f'{v:>9.3f}±{e:<4.3f}'
                                  for v, e in zip(vals, errs))
              + f'{k:>14.3f}')
    print('  (площадь ⟹ показатель 0.5; смесь с объёмом конуса ⟹ выше)')
    print()
    print('=' * 100)
    print('ИТОГ ВИЛКИ')
    print(f'''
Сигнал B2: dA/A = −(1/60)τ²R_uu = 0.17 % при τ²R_uu = 0.1.
Нужно 3 σ, то есть σ(A)/A ≤ {100*TARGET:.3f} % на всю выборку.
Число ромбов читается в последнем столбце первой таблицы.
''')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
