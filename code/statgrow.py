#!/usr/bin/env python3
"""РАСТУЩИЕ МИРЫ: выбор по ходу, а не вес на готовом.

ПОЧЕМУ ИМЕННО ТАК. `preclusion-doc.md` вывел оценку: вес, зависящий от
расстановки через один скаляр ГОТОВОГО мира, геометрию отбирать не может
— он лишь перевешивает распределение этого скаляра. Под неё одинаково
попадают exp(-beta*S), exp(iS) и предотвращение.

Произведение по шагам роста под оценку НЕ попадает: множитель зависит от
локального шага, а не от итогового числа. И это у нас считается — фаза
шага 0.998 против фазы мира 10^-304 (`step-phase-doc.md`), плюс
аддитивность действия по шагам до 4.6e-15. То есть процесс вычислим, а
его итог нет: отбор нельзя записать формулой на расстановке, его можно
только ЗАПУСТИТЬ.

ПРОЦЕСС. Мир растёт по одному событию. Новое событие максимально, его
прошлое — любое вниз-замкнутое подмножество. Классическое содержание
exp(iS) есть СТАЦИОНАРНОСТЬ, и она локальна, поэтому правило шага:

    выбрать прошлое, при котором |dS| минимально

Ни одного свободного параметра: ни температуры, ни порога, ни связи.
Приращение считается точно (аддитивность проверена): для нового
максимального j с прошлым P

    dS = 1 + sum_{x in P} w(n(x,j)),   n(x,j) = |{z in P : x < z}|
    w = [-1, +9, -16, +8] при n = 0,1,2,3 и 0 дальше

СЕМЕЙСТВА ДЛЯ СРАВНЕНИЯ, все на одних и тех же кандидатах:
    стац    — минимум |dS|            (проверяемое)
    случ    — равновероятно           (нулевой процесс)
    макс    — максимум |dS|           (обратное)
    мин     — минимум dS              (жадное понижение действия)
    макс+   — максимум dS

ЧЕСТНАЯ ОГОВОРКА. Всех вниз-замкнутых подмножеств экспоненциально много,
поэтому на каждом шаге берётся ВЫБОРКА кандидатов (прошлые случайных
событий, их объединения, пустое и полное). Это практическое ограничение,
а не свойство правила, и оно одинаково для всех семейств.

МЕРИМ выросшие миры критерием годности проекта (`PICTURE.md` §5).
Эталон конуса: доля пар 0.098-0.103, размерность 4.00, спектр > 0 и
растущий с N, связей на событие ~ N^0.75, Жаккар 0.169.

  python3 statgrow.py
  python3 statgrow.py --n 600 --cands 24 --seeds 6
"""
import argparse
import math
import time

import numpy as np

from core import frac_pairs, mm_dim, height, spectral, links_of
from core import transitivity_violations

W = np.array([-1.0, 9.0, -16.0, 8.0])


def dS_of(Pm, idx):
    """Приращение действия при добавлении максимального события.

    Pm[i,j] = True, если j раньше i (конвенция проекта).
    idx — индексы вниз-замкнутого прошлого нового события.
    """
    if len(idx) == 0:
        return 1.0
    sub = Pm[np.ix_(idx, idx)]
    n = sub.sum(0)                      # для каждого x: сколько z выше
    tot = 1.0
    for m in range(4):
        tot += W[m] * float((n == m).sum())
    return tot


def candidates(Pm, k, rs, ncand):
    """Выборка вниз-замкнутых подмножеств текущего мира (k событий)."""
    out = [np.empty(0, np.int64)]                    # новый минимальный
    if k > 0:
        out.append(np.arange(k))                     # новый верх
    for _ in range(ncand):
        m = rs.integers(1, 4)
        vs = rs.integers(0, k, m) if k > 0 else []
        mask = np.zeros(k, bool)
        for v in vs:
            mask[int(v)] = True
            mask |= Pm[int(v), :k]
        out.append(np.flatnonzero(mask))
    # убрать повторы по размеру и сумме (дешёвый ключ)
    seen, uniq = set(), []
    for c in out:
        key = (len(c), int(c.sum()) if len(c) else -1)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def sensitivity(Pm, k, D, rs, npert=4):
    """Чувствительность dS к варьированию выбора: это и есть delta S = 0.

    ПОЧЕМУ. Правило «минимум |dS|» выродилось: приращение равно РОВНО
    нулю на прошлом из одного события (1 + w(0) = 1 - 1 = 0), и жадность
    это находит, давая плоский лес. Но стационарность означает не
    «dS = 0», а «dS не чувствителен к варьированию выбора». Меряем
    разброс dS по одноэлементным возмущениям кандидата.
    """
    vals = [dS_of(Pm, D)]
    inD = np.zeros(k, bool)
    if len(D):
        inD[D] = True
    for _ in range(npert):
        if rs.random() < 0.5 or len(D) == 0:
            out = np.flatnonzero(~inD)
            if len(out) == 0:
                continue
            v = int(rs.choice(out))
            mask = inD.copy()
            mask[v] = True
            mask |= Pm[v, :k]
            vals.append(dS_of(Pm, np.flatnonzero(mask)))
        else:
            sub = Pm[np.ix_(D, D)]
            top = D[sub.sum(0) == 0]
            if len(top) == 0:
                continue
            v = int(rs.choice(top))
            mask = inD.copy()
            mask[v] = False
            vals.append(dS_of(Pm, np.flatnonzero(mask)))
    return float(np.std(vals))


def grow(N, rule, ncand, rs):
    """Вырастить мир из N событий данным правилом шага.

    Два правила с СОСТОЯНИЕМ (процесс, а не функция готового мира):

    средн — шаг выбирает прошлое, чьё dS ближе всего к СРЕДНЕМУ dS,
            набранному миром до сих пор. Параметра нет: цель — его же
            собственное прошлое поведение. Самосогласование вместо
            оптимума.

    фаза  — шаг выбирает dS, ближайшее к КРАТНОМУ 2*pi. Приращение
            действия целое, а 2*pi иррационально, поэтому фазово
            нейтральные шаги образуют редкий набор целых: 0, 6, 13, 19,
            25, 31. Мир из таких шагов имеет фазу порядка единицы, а не
            10^-304 — то есть это прямая атака на расфазировку,
            сделанная локально и без параметра.
    """
    Pm = np.zeros((N, N), bool)
    run = []
    for k in range(N):
        if k == 0:
            continue
        cs = candidates(Pm, k, rs, ncand)
        ds = np.array([dS_of(Pm, c) for c in cs])
        if rule == 'стац':
            pick = int(np.argmin(np.abs(ds)))
        elif rule == 'варьяц':
            sens = np.array([sensitivity(Pm, k, c, rs) for c in cs])
            pick = int(np.argmin(sens))
        elif rule == 'случ':
            pick = int(rs.integers(0, len(cs)))
        elif rule == 'средн':
            tgt = float(np.mean(run)) if run else 0.0
            pick = int(np.argmin(np.abs(ds - tgt)))
        elif rule == 'фаза':
            tp = 2.0 * math.pi
            pick = int(np.argmin(np.abs(ds - tp * np.round(ds / tp))))
        elif rule == 'макс':
            pick = int(np.argmax(np.abs(ds)))
        elif rule == 'мин':
            pick = int(np.argmin(ds))
        else:
            pick = int(np.argmax(ds))
        D = cs[pick]
        run.append(float(ds[pick]))
        if len(D):
            Pm[k, D] = True
            # транзитивное замыкание: прошлое прошлого
            Pm[k, :k] |= Pm[D, :k].any(0)
    return Pm


def report(R, lbl, N):
    f = frac_pairs(R)
    h = height(R)[0]
    L = links_of(R)
    lpe = L.sum() / N
    A = R.astype(np.float32)
    CP = A @ A.T
    sz = A.sum(1)
    mn = sz[:, None] + sz[None, :] - CP
    iu = np.triu_indices(N, 1)
    jac = float(np.mean(CP[iu] / np.maximum(mn[iu], 1.0)))
    print(f'{lbl:>10}{f:>10.4f}{mm_dim(f):>8.2f}{h:>8.0f}'
          f'{spectral(R):>+10.4f}{lpe:>9.2f}{jac:>9.4f}'
          f'{transitivity_violations(R):>8}', flush=True)
    return f, h, lpe


def main():
    ap = argparse.ArgumentParser(description='Растущие миры')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--cands', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--rules', nargs='+',
                    default=['варьяц', 'стац', 'случ'])
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n}, кандидатов на шаг ~{a.cands}, засевов {a.seeds}')
    print('ЭТАЛОН конуса: доля 0.098-0.103, разм. 4.00, спектр>0 и растёт,')
    print('              связей/событие ~ N^0.75, Жаккар 0.169, наруш. 0')
    print()
    print(f"{'правило':>10}{'доля пар':>10}{'разм.':>8}{'высота':>8}"
          f"{'спектр':>10}{'связ/соб':>9}{'Жаккар':>9}{'наруш':>8}")

    acc = {}
    for rule in a.rules:
        rows = []
        for t in range(a.seeds):
            rs = np.random.default_rng(3501000 + 977 * t)
            Pm = grow(a.n, rule, a.cands, rs)
            rows.append(report(Pm, f'{rule}/{t}', a.n))
            del Pm
        acc[rule] = np.array(rows)
        m = acc[rule].mean(0)
        print(f'{"среднее":>10}{m[0]:>10.4f}{mm_dim(m[0]):>8.2f}'
              f'{m[1]:>8.1f}{"":>10}{m[2]:>9.2f}')
        print()

    print('показатель высоты по N (эталон 0.279) — отдельным прогоном')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
