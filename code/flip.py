#!/usr/bin/env python3
"""ЕСТЬ ЛИ НАПРАВЛЕНИЕ, ГДЕ СЛОИСТЫЕ ШУМЯТ СИЛЬНЕЕ? Точный вопрос.

ЧТО БЫЛО НЕ ТАК В `design.py`. Я минимизировал `σ` многообразных — и
получил падение в 36 раз, устойчивое на отложенной половине. Но знак не
перевернулся, потому что у слоистых `σ` падала тоже: отношение
`s_м/s_л` стояло на 3…6 при любом `K`. Минимизировалась НЕ ТА величина.

ПРАВИЛЬНАЯ ПОСТАНОВКА. Нужен максимум РАЗНОСТИ `σ²_л − σ²_м`. При
линейных условиях на среднее это сводится к одному вопросу:

    существует ли направление `v` с `Aᵀv = 0`, для которого
    `vᵀΣ_л v > vᵀΣ_м v` ?

Если да — его можно добавлять с любым весом, и знак перевернётся: путь
открыт. Если нет, то `Σ_м − Σ_л` положительно определена на нулевом
подпространстве условий, и путь закрыт СТРУКТУРНО, а не из-за неудачного
выбора комбинации. Ответ даёт максимальное обобщённое собственное число
пары `(Σ_л, Σ_м)` на этом подпространстве:

    λ_max > 1  →  открыт,   λ_max ≤ 1  →  закрыт.

ПОЧЕМУ ПРЕЖНИЙ ЗАМЕР НЕ ГОДИЛСЯ. Слоистых было 20 реализаций на 17
измерений — ковариация почти вырождена, и обобщённые числа вышли бы
мусором. Здесь по 200 на сектор.

ЗАЩИТА ОТ ПОДГОНКИ. Направление строится на ПОЛОВИНЕ засевов, а
`σ_л/σ_м` в нём считается на ОТЛОЖЕННОЙ. Собственный вектор почти
вырожденной пары — классический пожиратель шума, поэтому смотреть надо
только отложенный столбец.

  python3 flip.py
"""
import math
import time

import numpy as np

from sprinkle import sprinkle_d
from uphill import gen3
from design import vec, bd_w, curved_sample, KMAX, PARTS


def collect(gen, n, seeds, root, **kw):
    out = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 97 * s), **kw)
        if isinstance(A, tuple):
            A = A[0]
        out.append(vec(A))
        del A
    return np.array(out, float)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N1, N2, SD = 800, 1600, 200
    print('Сбор выборок по', SD, 'на сектор', flush=True)
    M1 = collect(sprinkle_d, N1, 60, 21000, d=4)
    MAN = collect(sprinkle_d, N2, SD, 22000, d=4)
    print(f'  многообразные: {len(MAN)} ({time.time()-t0:.0f} с)', flush=True)
    LAY = collect(gen3, N2, SD, 23000, parts=PARTS, q=0.5)
    print(f'  слоистые: {len(LAY)} ({time.time()-t0:.0f} с)', flush=True)
    Cp, Rp = curved_sample(+0.10, 30, 24000)
    Cm, Rm = curved_sample(-0.10, 30, 25000)
    dmu = Cp.mean(axis=0) - Cm.mean(axis=0)
    dR = float(Rp.mean() - Rm.mean())
    print(f'  кривые ({time.time()-t0:.0f} с)', flush=True)
    print()

    h = SD // 2
    trM, teM = MAN[:h], MAN[h:]
    trL, teL = LAY[:h], LAY[h:]
    A = np.column_stack([M1.mean(axis=0), trM.mean(axis=0), dmu])

    # нулевое подпространство условий
    U, S_, Vt = np.linalg.svd(A.T)
    Z = Vt[len(S_):].T                      # (KMAX+1) x (KMAX+1-3)
    print('=' * 92)
    print(f'Нулевое подпространство условий: размерность {Z.shape[1]}'
          f' из {KMAX+1}')
    print()

    def cov(X):
        return np.cov((X - X.mean(axis=0)).T)

    SM = Z.T @ cov(trM) @ Z
    SL = Z.T @ cov(trL) @ Z
    # регуляризация по следу, чтобы вырождение не выдавало себя за сигнал
    reg = 1e-9 * np.trace(SM) / SM.shape[0]
    SMr = SM + reg * np.eye(SM.shape[0])
    ev = np.linalg.eigvals(np.linalg.solve(SMr, SL))
    ev = np.sort(np.real(ev[np.abs(np.imag(ev)) < 1e-6]))[::-1]
    print('ОБОБЩЁННЫЕ СОБСТВЕННЫЕ ЧИСЛА λ = (σ_л/σ_м)² по направлениям')
    print('  ' + '  '.join(f'{v:.3g}' for v in ev[:8]))
    print(f'  λ_max = {ev[0]:.4g}   (нужно > 1)')
    print()
    print('=' * 92)
    print('ПРОВЕРКА НА ОТЛОЖЕННОЙ: правда ли в лучшем направлении так?')
    print(f"{'направление':>26}{'σ_л/σ_м обуч.':>16}{'σ_л/σ_м ОТЛОЖ.':>17}"
          f"{'вывод':>12}")
    w_, v_ = np.linalg.eig(np.linalg.solve(SMr, SL))
    o = np.argsort(-np.real(w_))
    for rank in range(min(3, len(o))):
        v = Z @ np.real(v_[:, o[rank]])
        r_tr = np.std(trL @ v, ddof=1) / max(np.std(trM @ v, ddof=1), 1e-30)
        r_te = np.std(teL @ v, ddof=1) / max(np.std(teM @ v, ddof=1), 1e-30)
        print(f'{"собств. #" + str(rank + 1):>26}{r_tr:>16.4f}{r_te:>17.4f}'
              f'{("ДА" if r_te > 1 else "нет"):>12}')
    bd = bd_w()
    r_bd = np.std(teL @ bd, ddof=1) / np.std(teM @ bd, ddof=1)
    print(f'{"БД (для сравнения)":>26}{"":>16}{r_bd:>17.4f}'
          f'{("ДА" if r_bd > 1 else "нет"):>12}')
    print()
    print('=' * 92)
    best = max(np.std(teL @ (Z @ np.real(v_[:, o[r]])), ddof=1)
               / max(np.std(teM @ (Z @ np.real(v_[:, o[r]])), ddof=1), 1e-30)
               for r in range(min(3, len(o))))
    if best > 1.0:
        print(f'''ОТКРЫТ. На отложенной половине нашлось направление с
   σ_л/σ_м = {best:.3f} > 1, совместимое со всеми условиями на среднее.
   Значит разность `σ²_л − σ²_м` можно сделать положительной, знак
   переворачивается, и полоса по `ℏ` существует.''')
    else:
        print(f'''ЗАКРЫТ СТРУКТУРНО. Лучшее отложенное отношение
   σ_л/σ_м = {best:.3f} ≤ 1, то есть `Σ_м − Σ_л` положительно определена
   на всём нулевом подпространстве условий. Ни одна линейная комбинация
   счётов слоёв с верным непрерывным средним не даёт слоистым шуметь
   сильнее многообразных. Это свойство самих счётов, а не выбора
   действия.''')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
