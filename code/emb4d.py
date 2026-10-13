#!/usr/bin/env python3
"""ОБЪЁМ ВЛОЖЕНИЙ: есть ли в 4d место для динамики, а в 1+1 нет.

Оба вопроса оказываются одним. Мера засева на классе равна его ОБЪЁМУ в
пространстве конфигураций (в 1+1 это число реализаторов). Если объём у
всех классов одинаков, равномерная мера и ЕСТЬ засев, и динамике взяться
негде. Если объём сильно разный, засев есть равномерная мера С ВЕСОМ, и
этот вес ln(объём) и есть действие - не фундаментальное, а энтропия
вложений.

В 1+1 (`kin11-doc.md`) объём варьируется (кратность до 40 при n=9), но
расхождение мер всего 4.0% и тает как n^{-1.72}.

ПРЕДСКАЗАНИЕ для 4d, до замера: объём обязан варьироваться несравнимо
сильнее - антицепь в 4d почти неизбежна (пространственноподобность
типична), длинная цепь крайне маловероятна. Если расхождение велико и НЕ
тает, ответ на оба вопроса один: динамика в 4d есть неравномерность
объёма вложений, а в 1+1 её нет.

КОНТРОЛЬ МЕТОДИКИ, обязательный: классы в 4d набираются ВЫБОРКОЙ, а
выборка пропускает редкие классы и смещает равномерную меру. Поэтому то
же самое считается выборкой для 1+1, где точный ответ известен
(расхождение r: 8.1% при n=6, 6.7% при n=7). Если выборочная оценка его
воспроизводит - методике верим.
"""
import math

import numpy as np

from perm2iso import canon

KEYS = ('r', 'H', 'n0', 'n1', 'n2', 'n3')


def pts_4d(m, n, rs):
    """m групп по n точек в causal diamond размерности 4."""
    need = m * n
    out = []
    while sum(len(o) for o in out) < need:
        k = int((need) * 3) + 1000
        t = rs.random(k)
        v = rs.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = rs.random(k) ** (1.0 / 3.0) * 0.5
        ok = rad < np.minimum(t, 1 - t)
        P = np.hstack([t[ok, None], v[ok] * rad[ok, None]])
        out.append(P)
    P = np.vstack(out)[:need]
    return P.reshape(m, n, 4)


def pts_11(m, n, rs):
    return rs.random((m, n, 2))


def order_4d(P):
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt ** 2 > dx2)


def order_11(P):
    u, v = P[:, 0], P[:, 1]
    return (u[None, :] > u[:, None]) & (v[None, :] > v[:, None])


def stats(R):
    n = R.shape[0]
    Iv = R.astype(np.int16) @ R.astype(np.int16)
    sz = Iv[R]
    h = np.zeros(n, int)
    for x in np.argsort(R.sum(1))[::-1]:
        pre = np.nonzero(R[:, x])[0]
        h[x] = (h[pre].max() + 1) if len(pre) else 1
    return dict(r=R.sum() / (n * (n - 1) / 2), H=int(h.max()),
                n0=int((sz == 0).sum()), n1=int((sz == 1).sum()),
                n2=int((sz == 2).sum()), n3=int((sz == 3).sum()))


def run(n, m, maker, order, rs):
    tally, acc = {}, {}
    half = None
    for i, P in enumerate(maker(m, n, rs)):
        R = order(P)
        k = canon(R)
        if k is None:
            continue
        tally[k] = tally.get(k, 0) + 1
        if k not in acc:
            acc[k] = stats(R)
        if i + 1 == m // 2:
            half = len(tally)
    tot, d = sum(tally.values()), len(tally)
    a = {k: sum(tally[x] * acc[x][k] for x in tally) / tot for k in KEYS}
    b = {k: sum(acc[x][k] for x in tally) / d for k in KEYS}
    mult = np.array(list(tally.values()), float) / tot
    return dict(d=d, half=half, a=a, b=b,
                spread=float(mult.max() / mult.min()),
                tot=tot)


def main():
    print(__doc__)
    rs = np.random.default_rng(777111)
    M = 40000
    print('КОНТРОЛЬ МЕТОДИКИ: выборка в 1+1 против точного ответа')
    print(f"{'n':>3}{'классов':>9}{'на половине':>13}"
          f"{'расх. r (выб.)':>16}{'точно':>8}")
    exact = {6: 8.1, 7: 6.7, 8: 5.2}
    for n in (6, 7):
        r = run(n, M, pts_11, order_11, rs)
        dr = (r['b']['r'] - r['a']['r']) / r['a']['r'] * 100
        print(f'{n:>3}{r["d"]:>9}{r["half"]:>13}{dr:>15.1f}%'
              f'{exact[n]:>7.1f}%', flush=True)

    print('\n4d: тот же счёт')
    print(f"{'n':>3}{'классов':>9}{'на половине':>13}{'разброс объёма':>16}"
          + ''.join(f'{f"расх. {k}":>12}' for k in ('r', 'H', 'n0')))
    for n in (5, 6, 7):
        r = run(n, M, pts_4d, order_4d, rs)
        ds = [(r['b'][k] - r['a'][k]) / max(abs(r['a'][k]), 1e-9) * 100
              for k in ('r', 'H', 'n0')]
        print(f'{n:>3}{r["d"]:>9}{r["half"]:>13}'
              f'{r["spread"]:>16.0f}'
              + ''.join(f'{v:>11.1f}%' for v in ds), flush=True)
    print('\nЧТЕНИЕ: «на половине» близко к «классов» => выборка насытилась.')
    print('Разброс объёма - отношение самого частого класса к самому редкому.')


if __name__ == '__main__':
    main()
