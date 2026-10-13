#!/usr/bin/env python3
"""ФУНКЦИЯ МЁБИУСА: различает ли она секторы.

Обзор механизмов гашения (11 окт.) дал один общий вывод: огромная
когерентная сумма схлопывается, когда у неё есть ЗНАК, привязанный к
комбинаторному инварианту конфигурации — чётности перестановки
(фермионные определители), числу намотки (тета-вакуумы), порядку
(вандермонд). А не к счёту чего-либо. Наши же действия типа БД зависят
ТОЛЬКО от счётов N_m, и потому пол 0.056 не пробивался.

Функция Мёбиуса порядка — ровно такой инвариант. Она определена
условием sum_{x<=z<=y} mu(x,z) = delta_{x,y}, то есть СОЗДАНА для
схлопывания сумм, и зависит от ФОРМЫ интервала, а не от размера:
  * связь (пустой интервал)         -> mu = -1
  * интервал-антицепь из m элементов -> mu = m - 1
  * в общем случае — приведённая эйлерова характеристика комплекса
    порядка открытого интервала.
Это вне всех закрытых классов по построению.

СЧЁТ ТОЧНЫЙ: mu есть матрица, обратная зета-матрице (Z[x,y]=1 при
x<=y). В порядке линейного расширения Z верхнетреугольная с единицами
на диагонали, значит обращается целочисленно без приближений.

ПРЕДРЕГИСТРАЦИЯ. У слоистых интервалы простые (связь либо антицепь
соседнего слоя), значит mu велика и регулярна. У засева интервалы
конусные, значит mu прыгает. Решает степень ВЗАИМНОГО ГАШЕНИЯ
|sum mu| / sum |mu|: если у слоистых она много меньше, чем у засева,
то амплитуда с весом Мёбиуса подавляет слоистых — то есть то, что нам
и нужно. Если наоборот — идея умирает сразу.

КОНТРОЛИ:
  A. Z @ M = I точно (целочисленно);
  B. mu на связях обязана быть -1 ровно у всех;
  C. mu на интервалах-антицепях обязана быть m-1 (проверяется на
     слоистом секторе, где такие интервалы есть по построению).
"""
import math

import numpy as np

from qcarrier import sprinkle4


def mobius(Rc):
    """mu как обратная зета-матрица; точно, в целых."""
    n = Rc.shape[0]
    # порядок линейного расширения: по числу предков
    key = Rc.sum(axis=0)
    order = np.argsort(key, kind='stable')
    R = Rc[np.ix_(order, order)]
    Z = (R | np.eye(n, dtype=bool)).astype(np.int64)
    if not np.all(np.triu(Z) == Z):
        raise RuntimeError('не верхнетреугольная — порядок неверен')
    M = np.zeros((n, n), np.int64)
    np.fill_diagonal(M, 1)
    for j in range(n):
        for i in range(j - 1, -1, -1):
            if Z[i, j]:
                M[i, j] = -int(Z[i, i + 1:j + 1] @ M[i + 1:j + 1, j])
            elif (Z[i, i + 1:j] @ M[i + 1:j, j]) != 0:
                M[i, j] = -int(Z[i, i + 1:j + 1] @ M[i + 1:j + 1, j])
    return R, M


def stats(Rc, name, check_c=False):
    R, M = mobius(Rc)
    n = R.shape[0]
    ident = np.abs((R | np.eye(n, dtype=bool)).astype(np.int64) @ M
                   - np.eye(n, dtype=np.int64)).max()
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int64)
    links = R & (Iv == 0)
    mu_links = M[links]
    rel = R
    mu = M[rel]
    tot = int(mu.sum())
    abstot = int(np.abs(mu).sum())
    out = dict(n=n, ident=int(ident), R=int(rel.sum()),
               bad_links=int((mu_links != -1).sum()),
               tot=tot, abstot=abstot,
               ratio=abs(tot) / max(abstot, 1),
               maxabs=int(np.abs(mu).max()),
               mean_abs=float(np.abs(mu).mean()))
    if check_c:
        # интервалы-антицепи: mu обязана быть m-1
        anti, bad = 0, 0
        xs, ys = np.nonzero(rel)
        for x, y in zip(xs[:4000], ys[:4000]):
            mid = np.nonzero(R[x] & R[:, y])[0]
            if len(mid) >= 1 and not R[np.ix_(mid, mid)].any():
                anti += 1
                if M[x, y] != len(mid) - 1:
                    bad += 1
        out['anti'] = anti
        out['anti_bad'] = bad
    return out


def layered(n, k, q, rs, thin=None):
    a = n // k
    sz = [a] * k
    sz[-1] += n - a * k
    if thin and k >= 3 and a > thin:
        sz = [a] * k
        sz[0] = sz[-1] = thin
        sz[k // 2] += n - sum(sz)
    off = np.cumsum([0] + sz)
    Rc = np.zeros((n, n), bool)
    for i in range(k - 1):
        Rc[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = (
            rs.random((sz[i], sz[i + 1])) < q)
    C = Rc.copy()
    for _ in range(k):
        nx = C | ((C.astype(np.float32) @ C.astype(np.float32)) > .5)
        if (nx == C).all():
            break
        C = nx
    return C


def main():
    print(__doc__)
    rs = np.random.default_rng(31415)

    print('КОНТРОЛИ на одном случае каждого вида (N=200)')
    for nm, Rc in (('засев d=4', None),
                   ('слоистый k=3 q=0.2', None)):
        Rc = (sprinkle4(200, rs, 1.0) if nm.startswith('засев')
              else None)
        if Rc is not None:
            P = Rc
            t, X = P[:, 0], P[:, 1:]
            dt = t[None, :] - t[:, None]
            d = X[None, :, :] - X[:, None, :]
            Rc = (dt > 0) & (dt * dt > (d * d).sum(-1))
        else:
            Rc = layered(200, 3, 0.2, rs)
        s = stats(Rc, nm, check_c=True)
        print(f'  {nm:>20}: Z@M-I = {s["ident"]}  '
              f'связей с mu != -1: {s["bad_links"]}  '
              f'интервалов-антицепей {s["anti"]}, из них неверных '
              f'{s["anti_bad"]}')

    print(f"\n{'сектор':>22}{'N':>6}{'отношений':>11}{'sum mu':>12}"
          f"{'sum |mu|':>12}{'гашение':>10}{'max|mu|':>9}")
    rows = {}
    for n in (200, 400, 800):
        P = sprinkle4(n, rs, 1.0)
        t, X = P[:, 0], P[:, 1:]
        dt = t[None, :] - t[:, None]
        d = X[None, :, :] - X[:, None, :]
        s = stats((dt > 0) & (dt * dt > (d * d).sum(-1)), 'засев')
        rows.setdefault('засев d=4', []).append(s)
        print(f'{"засев d=4":>22}{n:>6}{s["R"]:>11}{s["tot"]:>12}'
              f'{s["abstot"]:>12}{s["ratio"]:>10.2e}{s["maxabs"]:>9}',
              flush=True)
    for lab, k, q, thin in (('слоистый k=3 q=0.2', 3, 0.2, None),
                            ('слоистый k=4 тонкий', 4, 0.15, 2),
                            ('слоистый k=5 q=0.1', 5, 0.1, None)):
        for n in (200, 400, 800):
            s = stats(layered(n, k, q, rs, thin), lab)
            rows.setdefault(lab, []).append(s)
            print(f'{lab:>22}{n:>6}{s["R"]:>11}{s["tot"]:>12}'
                  f'{s["abstot"]:>12}{s["ratio"]:>10.2e}'
                  f'{s["maxabs"]:>9}', flush=True)

    print('\nПОКАЗАТЕЛИ по N (200 -> 800)')
    for lab, ss in rows.items():
        ea = math.log(max(ss[-1]['abstot'], 1)
                      / max(ss[0]['abstot'], 1)) / math.log(4)
        print(f'  {lab:>22}: sum|mu| ~ N^{ea:.3f}, '
              f'гашение {ss[0]["ratio"]:.1e} -> {ss[-1]["ratio"]:.1e}')
    print('\nЧТЕНИЕ: если у слоистых гашение много сильнее (отношение')
    print('много меньше), амплитуда с весом Мёбиуса их подавляет.')


if __name__ == '__main__':
    main()
