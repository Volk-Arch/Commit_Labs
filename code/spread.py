#!/usr/bin/env python3
"""РАЗБРОС как второй канал: есть ли у кривизны вторая компонента.

ЗАЧЕМ. delay-doc.md закончился резкой формулировкой зазора: у нас ход
часов и предпочтение путей — ОДНА величина (плотность звеньев вдоль
пути), поэтому добавленная плотность ускоряет часы и притягивает, а
убранная замедляет и отталкивает. Ни один вариант не даёт двух нужных
знаков сразу. В ОТО они разные, потому что g00 и пространственная часть
метрики — разные компоненты.

Значит нужна ВТОРАЯ независимая величина. Архив утверждает, что порядок
помнит о входе ровно две симметричные функции: СУММУ шкал (R2 = 0.948,
это время) и РАЗБРОС (R2 = 0.909), ортогональные (корреляция -0.01), и
третьей нет. Вся геометрия до сих пор строилась на сумме. Разброс в
кривизну не входил ни разу.

ОГОВОРКА, БЕЗ КОТОРОЙ НЕЛЬЗЯ. Разброс самих ЗНАЧЕНИЙ шкал убит теоремой
о перешкалировании: на одном и том же порядке он подскакивает с 0.064 до
1096. Поэтому берётся не он, а ПОРЯДКОВЫЙ аналог — величина, считаемая
из одного отношения и потому инвариантная:

    для события x взять его прямые предшественники L(x) и померить,
    насколько они разнесены ДРУГ ОТ ДРУГА мерой общего прошлого

        spread(x) = 1 - <q(y,y')>  по парам y,y' из L(x),
        q(y,y') = |past(y) & past(y')| / min(|past(y)|,|past(y')|)

    Большой spread = широкая «пасть» прошлого, малый = узкая.

Это второй канал: он не о том, СКОЛЬКО звеньев (это h), и не о том,
сколько событий (это V), а о том, насколько РАЗВЕДЕНЫ направления, по
которым событие получает своё прошлое.

ЧТО РЕШАЕТ ОПЫТ. Оба канала меряются у ДВУХ объектов сразу:

    дырка  (убранная плотность): часы замедляет, цепи отталкивает
    трубка (добавленная, лестница): часы ускоряет, цепи притягивает

    если spread отвечает ТАК ЖЕ, как часы, у обоих — второй компоненты
        нет, и зазор остаётся;
    если spread различает дырку и трубку ИНАЧЕ, чем часы — есть две
        независимые компоненты, и расцепление возможно.

Контроль у каждого объекта — удаление или добавление того же числа
событий случайно по ромбу.

  python3 spread.py
  python3 spread.py --rad 0.24 --mass 240 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from weight import diamond, cone_rel
from delay import heights, chain_tube


def links_of(R):
    """Прямые связи: связь есть и интервал пуст."""
    A = np.asarray(R, np.float32)
    return np.asarray(R & ((A @ A) == 0), bool)


def channels(X, minlinks=4):
    """Два канала на каждом событии: ход часов и разброс пасти."""
    R = cone_rel(X)
    h = heights(R).astype(np.float64)
    V = R.sum(1).astype(np.float64)
    L = links_of(R)
    A = R.astype(np.float32)
    CP = A @ A.T
    sz = A.sum(1)
    mn = np.minimum(sz[:, None], sz[None, :])
    q = CP / np.maximum(mn, 1.0)
    sp = np.full(len(X), np.nan)
    for i in range(len(X)):
        pre = np.flatnonzero(L[i])
        pre = pre[sz[pre] >= 4]
        if len(pre) < minlinks:
            continue
        sub = q[np.ix_(pre, pre)]
        iu = np.triu_indices(len(pre), 1)
        sp[i] = 1.0 - float(sub[iu].mean())
    return h, V, sp


def profile(X, T, edges, ced):
    """Остаток log h по V и разброс, разложенные по радиусу."""
    h, V, sp = channels(X)
    ok = (V >= 40) & (V <= 600) & (h > 0)
    r = np.linalg.norm(X[:, 1:], axis=1)
    out_c = np.full(len(edges) - 1, np.nan)
    out_s = np.full(len(edges) - 1, np.nan)
    if ok.sum() > 60:
        lv, lh = np.log(V[ok]), np.log(h[ok])
        res = lh - np.polyval(np.polyfit(lv, lh, 1), lv)
        rr = r[ok]
        for b in range(len(edges) - 1):
            m = (rr >= edges[b]) & (rr < edges[b + 1])
            if m.sum() > 5:
                out_c[b] = res[m].mean()
    oks = np.isfinite(sp) & (V >= 40) & (V <= 600)
    if oks.sum() > 60:
        # разброс тоже чистим от зависимости от V
        lv = np.log(V[oks])
        rs_ = sp[oks] - np.polyval(np.polyfit(lv, sp[oks], 1), lv)
        rr = r[oks]
        for b in range(len(edges) - 1):
            m = (rr >= edges[b]) & (rr < edges[b + 1])
            if m.sum() > 5:
                out_s[b] = rs_[m].mean()
    return out_c, out_s


def main():
    ap = argparse.ArgumentParser(description='Разброс как второй канал')
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rad', type=float, default=0.24)
    ap.add_argument('--mass', type=int, default=240)
    ap.add_argument('--tuberad', type=float, default=0.02)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    ed = np.array([0.0, 0.18, 0.30, 0.45, 0.65, 0.90])
    print(f'ромб T={a.t}; два канала на событии')
    print('  ЧАСЫ    = остаток log h после снятия log V')
    print('  РАЗБРОС = 1 - <q> по парам прямых предшественников, '
          'тоже чищен от V')
    print(f'объекты: дырка радиуса {a.rad} и трубка из {a.mass} событий')
    print('у каждого контроль со тем же числом событий, '
          'рассыпанных по ромбу')
    print(f'засевов {a.seeds}')

    for kind in ('дырка', 'трубка'):
        print(f'\nОБЪЕКТ: {kind}')
        print(f"{'':>14}" + ''.join(f'{e:>12.2f}' for e in ed[1:]))
        acc = {}
        for mk in (True, False):
            C, S = [], []
            for t in range(a.seeds):
                rs = np.random.default_rng(2401000 + 977 * t)
                X0 = diamond(a.t, rs)
                if kind == 'дырка':
                    keep = np.linalg.norm(X0[:, 1:], axis=1) >= a.rad
                    k = int((~keep).sum())
                    if mk:
                        X = X0[keep]
                    else:
                        ii = rs.choice(len(X0), len(X0) - k, replace=False)
                        X = X0[np.sort(ii)]
                else:
                    ob = (chain_tube(a.mass, a.tuberad, a.t, rs) if mk
                          else diamond(a.t, rs, n_at_t1=a.mass / a.t ** 4))
                    X = np.vstack([ob, X0])
                c, s = profile(X, a.t, ed, None)
                C.append(c); S.append(s)
            C = np.array(C); S = np.array(S)
            acc[mk] = (np.nanmean(C, 0), np.nanstd(C, 0) / math.sqrt(len(C)),
                       np.nanmean(S, 0), np.nanstd(S, 0) / math.sqrt(len(S)))
            nm = 'объект' if mk else 'контроль'
            print(f'{nm+" часы":>14}' + ''.join(f'{x:>+12.5f}'
                                                for x in acc[mk][0]))
            print(f'{nm+" разбр":>14}' + ''.join(f'{x:>+12.5f}'
                                                 for x in acc[mk][2]))
        for j, nm in ((0, 'часы'), (2, 'разброс')):
            d = acc[True][j] - acc[False][j]
            er = np.hypot(acc[True][j + 1], acc[False][j + 1])
            print(f'{"разн. "+nm:>14}' + ''.join(f'{x:>+12.5f}' for x in d))
            print(f'{"значим.":>14}'
                  + ''.join(f'{(abs(x)/y if y > 1e-12 else float("nan")):>12.1f}'
                            for x, y in zip(d, er)), flush=True)

    print('\n  если знаки РАЗБРОСА у дырки и трубки повторяют знаки ЧАСОВ —')
    print('  второй компоненты нет; если различают иначе — расцепление есть')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
