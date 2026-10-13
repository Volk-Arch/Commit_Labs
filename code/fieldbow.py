#!/usr/bin/env python3
"""Загиб геодезических к полю: тяготение или погоня за звеньями.

ПЕРЕПРОВЕРКА архивного результата `gravity-doc.md`: «геодезические
загибаются к области», сдвиг радиуса −0.0154 ± 0.0021, z = 7.52, все
четыре реализации отрицательные.

ЧТО ТАМ БЫЛО ПОЛЕМ. Не добавленные события, а локально изменённое
ПРАВИЛО: у события свой набор активных шкал, сравнение идёт по шкалам,
активным у ОБОИХ. Около центра одна шкала выключена.

ПОДОЗРЕНИЕ. Из собственной таблицы архива: внутри доля пар 0.1041 против
0.0615 снаружи, связей на событие 194 против 118. Выключенная шкала
ОСЛАБЛЯЕТ условие, значит внутри связей больше, значит там доступны
более длинные цепи. А максимальная цепь по определению ищет, где звеньев
больше. Тогда загиб есть кинематика — «цепь идёт туда, где звеньев
гуще», — а не выведенное притяжение.

Сегодняшний диагноз лестницы (`delay-doc.md` §1) — то же самое в другом
исполнении: трубка добавленных событий имеет внутреннюю высоту 57…105
против фоновых 11.8 и притягивает цепи на 24σ.

РЕШАЮЩИЙ ХОД. Выключить шкалу не ВНУТРИ, а СНАРУЖИ. Тогда внутренняя
область становится относительно СТРОЖЕ, связей там меньше. Правило то
же, код тот же, меняется только кому досталась выключенная шкала.

    загиб кинематический  =>  знак сдвига ПЕРЕВЕРНЁТСЯ
    загиб от тяготения    =>  знак сохранится

Контроль, как в архиве: без поля вовсе сдвиг обязан быть ровно нулём
(те же пары концов, тот же мир).

  python3 fieldbow.py
  python3 fieldbow.py --k 6 --rad 0.5 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from detour import longest_between
from bow import bow_of


def diamond(N, T, rs):
    out, got = [], 0
    while got < N:
        m = int((N - got) * 9) + 64
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        ok = (Y[:, 0] >= r) & (T - Y[:, 0] >= r)
        if ok.any():
            out.append(Y[ok])
            got += int(ok.sum())
    X = np.vstack(out)[:N]
    return X[np.argsort(X[:, 0])]


def rel_active(S, A):
    """R[i,j] = i раньше j по всем шкалам, активным У ОБОИХ.

    S: (N,K) значения шкал. A: (N,K) булева активность.
    """
    N, K = S.shape
    R = np.ones((N, N), bool)
    for c in range(K):
        a = A[:, c]
        both = np.outer(a, a)
        less = S[None, :, c] > S[:, None, c]      # S[i,c] < S[j,c] -> R[j,i]
        R &= (~both.T) | less.T
    np.fill_diagonal(R, False)
    return R


def build(X, K, rad, where, rs):
    """Порядок при выключенной шкале 0 внутри / снаружи / нигде.

    Возвращает (R, маска внутри). Конвенция проекта: R[i,j] = «j раньше
    i» — та же, что у cone_rel в weight.py, и её ждут bow_of и
    longest_between. rel_active строит именно её, поэтому НИКАКОГО
    транспонирования на выходе: первая версия транспонировала, все связи
    смотрели назад, и пар не находилось вовсе.
    """
    u = rs.normal(size=(K, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    S = X[:, 0:1] - X[:, 1:] @ u.T
    inside = np.linalg.norm(X[:, 1:], axis=1) < rad
    A = np.ones((len(X), K), bool)
    if where.startswith('внутри'):
        m = int(where[-1]) if where[-1].isdigit() else 1
        A[np.ix_(inside, np.arange(m))] = False
    elif where == 'снаружи':
        A[~inside, 0] = False
    R = rel_active(S, A)
    return np.asarray(R, bool), inside


def viol(R, tries=60000, rs=None):
    """Доля нарушений транзитивности на выборке троек."""
    N = len(R)
    rs = rs or np.random.default_rng(0)
    i = rs.integers(0, N, tries); j = rs.integers(0, N, tries)
    k = rs.integers(0, N, tries)
    m = R[j, i] & R[k, j]
    if m.sum() == 0:
        return 0.0
    return float((~R[k, i][m]).mean())


def pairs_ring(X, T, d, tol, rs, npair=90):
    r = np.linalg.norm(X[:, 1:], axis=1)
    band = (np.abs(r - d) < tol) & (r > 1e-9)
    lo = np.flatnonzero(band & (X[:, 0] < 0.40 * T))
    hi = np.flatnonzero(band & (X[:, 0] > 0.60 * T))
    if len(lo) < 3 or len(hi) < 3:
        return []
    out = []
    for _ in range(npair * 18):
        if len(out) >= npair:
            break
        a, b = int(rs.choice(lo)), int(rs.choice(hi))
        ua = X[a, 1:] / np.linalg.norm(X[a, 1:])
        ub = X[b, 1:] / np.linalg.norm(X[b, 1:])
        if float(ua @ ub) < 0.85:
            continue
        out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser(description='Перепроверка загиба к полю')
    ap.add_argument('--n', type=int, default=2200)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--rad', type=float, default=0.50)
    ap.add_argument('--d', type=float, default=0.62)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    print(f'ромб T={a.t}, N={a.n}, шкал {a.k}; поле — выключенная шкала '
          f'в радиусе {a.rad}')
    print(f'пары концов на радиусе {a.d}, разнесены во времени; '
          f'мерим средний радиус максимальной цепи')
    print('ОТРИЦАТЕЛЬНЫЙ сдвиг = цепь заходит внутрь (архивный знак)')
    print(f'засевов {a.seeds}')
    print()
    print(f"{'поле':>10}{'прирост внутр':>15}{'снаружи':>10}"
          f"{'пар':>6}{'сдвиг радиуса':>15}{'±':>9}{'z':>7}"
          f"{'длина цепи':>12}{'без поля':>10}{'наруш':>8}")

    for where in ('внутри1', 'внутри2', 'внутри3',
                  'снаружи', 'нигде'):
        SH, LF, LB, LIN, LOUT, VI, NP = [], [], [], [], [], [], 0
        for t in range(a.seeds):
            rs = np.random.default_rng(3401000 + 977 * t)
            X = diamond(a.n, a.t, rs)
            rs2 = np.random.default_rng(555 + t)
            Rf, ins = build(X, a.k, a.rad, where, rs2)
            rs3 = np.random.default_rng(555 + t)
            R0, _ = build(X, a.k, a.rad, 'нигде', rs3)
            gi = (float(Rf[ins].sum(1).mean()) - float(R0[ins].sum(1).mean())
                  if ins.any() else np.nan)
            go = float(Rf[~ins].sum(1).mean()) - float(R0[~ins].sum(1).mean())
            LIN.append(gi); LOUT.append(go)
            VI.append(viol(Rf, rs=np.random.default_rng(7 + t)))
            ps = pairs_ring(X, a.t, a.d, 0.06, rs)
            sh, lf, lb = [], [], []
            for p, q in ps:
                vf = bow_of(Rf, X, p, q)
                vb = bow_of(R0, X, p, q)
                if vf is None or vb is None:
                    continue
                sh.append(vf - vb)
                lf.append(longest_between(Rf, p, q))
                lb.append(longest_between(R0, p, q))
            if sh:
                SH.append(float(np.mean(sh)))
                LF.append(float(np.mean(lf))); LB.append(float(np.mean(lb)))
                NP += len(sh)
            del Rf, R0, X
        if not SH:
            print(f'{where:>10}{"нет пар":>15}')
            continue
        S = np.array(SH)
        se = S.std() / math.sqrt(len(S))
        print(f'{where:>10}{np.nanmean(LIN):>15.1f}{np.mean(LOUT):>10.1f}'
              f'{NP:>6}{S.mean():>+15.5f}{se:>9.5f}'
              f'{abs(S.mean())/max(se,1e-12):>7.2f}'
              f'{np.mean(LF):>12.2f}{np.mean(LB):>10.2f}'
              f'{np.mean(VI):>8.4f}', flush=True)

    print('\n  знак перевернулся => загиб есть погоня за звеньями, '
          'не тяготение')
    print('  знак сохранился   => что-то сверх кинематики')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
