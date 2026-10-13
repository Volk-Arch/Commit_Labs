#!/usr/bin/env python3
"""ИГРА: одно локальное правило в каждой точке, линейка — ОТНОШЕНИЕ.

ЕГО МЫСЛЬ, с которой всё начинается. `gps-doc.md` уткнулся в
принципиальное: вещество ЕСТЬ разрежение, а собственное время считается
событиями (`|I|^(1/4)`, длина цепи — оба идут как `ρ^(1/4)`), поэтому
вещество меняет САМУ ЛИНЕЙКУ, которой мы мерим его влияние. Кривизна и
перекалибровка сцеплены по построению, и метрическая форма ОТО отверглась
на 11.9 сигм при том, что гипотеза сдвига плотности выполнилась на 0.3.

Его замечание: «меняя плотность, меняется линейка — а не может быть
линейка тоже относительной, то есть отношением чего-то к чему-то?»

Может, и это ровно то, чего не хватало. Проект с самого начала стоит на
том, что порядок осмыслен лишь с точностью до монотонного пересчёта —
значит и длина не вправе быть абсолютным счётом.

ПОСТРОЕНИЕ ЛИНЕЙКИ-ОТНОШЕНИЯ, выведенное, а не угаданное. В плоском
сприклинге плотности `ρ` интервал собственного времени `τ` содержит
`n = ρ(π/24)τ⁴` событий, а самая долгая цепь в нём имеет длину
`c = m₄(ρV)^(1/4) = m₄ n^(1/4)`. Пройдём от события назад ровно `k`
звеньев ПО САМОЙ ДОЛГОЙ ЦЕПИ — тогда `c = k` по построению, и

    k = m₄ · n^(1/4)      =>      n = (k / m₄)⁴

**Плотность сократилась тождественно.** Число событий в таком интервале
есть чистое число, одинаковое при любой плотности. Отклонение от него —
отклик, уже НЕ содержащий перекалибровки.

ПРАВИЛО ИГРЫ (одно, в каждой точке, ничего кроме порядка):

    1. из события x иди назад k звеньев по самой долгой цепи  -> предок a
    2. посчитай n(x) = число событий z с a < z < x
    3. линейка:  s(x) = k / n(x)^(1/4)

Всё локально (глубина `k`), всё счётом, и `s` — отношение, а не счёт.
Величина `s` в плоском мире равна `m₄` и НЕ меряется заранее, а берётся
из того же прогона без вещества: сравнивается всегда `s` с `s`.

СУДЬЯ. Проверка одна и решающая: отклик `s` у разрежения обязан быть
БОЛЬШЕ, чем у рассыпанного удаления того же числа событий. У прежней меры
`|I|^(1/4)` это отношение было всего 2.4 (и по форме совпадало с чистым
сдвигом плотности). Если у линейки-отношения рассыпанный контроль даст
НУЛЬ — затруднение снято, и разговор о кривизне становится осмысленным.

КОНТРОЛИ, поставленные до замера:
  1. рассыпанное удаление — главный судья (см. выше);
  2. нулевая дырка обязана дать ровно нуль;
  3. плотностная проверка НАПРЯМУЮ: тот же замер при ДВУХ разных `N` в
     одном и том же мире без вещества обязан дать одно и то же `s`,
     иначе линейка не плотностно-независима и всё дальнейшее напрасно;
  4. глубина `k` сканируется: ответ не должен от неё зависеть по сути;
  5. раскладка по радарному расстоянию от цепи-наблюдателя, то есть
     порядковая, а не координатная.

  python3 game.py                       # проверка плотности + отклик
  python3 game.py --n 20000 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from bigfall import build, radar_chain
from bend import heights_pred


def ruler(X, k, blk=1500, tlo=0.15, thi=0.95):
    """Линейка-отношение s(x) = k / n(x)^(1/4) в каждой точке.

    Возвращает s, признак определённости и `dmin` — наименьшее удаление
    от центра вещества по ВСЕМУ интервалу (включая его концы).

    `dmin` нужен для возражения, которое иначе снесло бы весь замер:
    при k = 5 локальный интервал охватывает около 4 ell собственного
    времени, то есть во внутренних полосах он ЗАДЕВАЕТ дырку напрямую, и
    отклик был бы не полем, а касанием. С `dmin` можно требовать, чтобы
    весь интервал лежал заведомо вне вещества.
    """
    n = len(X)
    h, pr = heights_pred(X, blk)
    t = np.ascontiguousarray(X[:, 0], np.float64)
    S = np.ascontiguousarray(X[:, 1:], np.float64)
    s = np.full(n, np.nan)
    dmin = np.full(n, np.nan)
    rad_all = np.linalg.norm(S, axis=1)
    # считаем только в полосе по времени: у краёв ромба нет ни прошлого,
    # ни будущего, и правило там не определено
    lo, hi = np.quantile(t, [tlo, thi])
    cand = np.flatnonzero((t >= lo) & (t <= hi) & (h >= k + 1))
    # УСКОРЕНИЕ. Локальный интервал крошечен по времени (у него глубина
    # всего k звеньев), поэтому перебирать весь мир незачем: события
    # отсортированы по времени, и нужный слой берётся двоичным поиском.
    # Это снимает множитель N с каждого события и делает достижимыми
    # миры радиусом свыше 20 ell, без которых чистого окна между защитой
    # от касания и границей мира не существует.
    order = np.argsort(t, kind='stable')
    ts = t[order]
    Ss = S[order]
    rs_all = np.linalg.norm(Ss, axis=1)
    for x in cand:
        a = x
        for _ in range(k):
            a = pr[a]
            if a < 0:
                break
        if a < 0:
            continue
        # n(x) = число событий в интервале (a, x) — только по слою
        lo_t, hi_t = t[a], t[x]
        i0 = np.searchsorted(ts, lo_t, 'right')
        i1 = np.searchsorted(ts, hi_t, 'left')
        if i1 <= i0:
            s[x] = k / 1.0
            dmin[x] = min(rad_all[a], rad_all[x])
            continue
        P = Ss[i0:i1]
        tp = ts[i0:i1]
        d1 = ((P - S[a]) ** 2).sum(1)
        d2 = ((S[x] - P) ** 2).sum(1)
        inside = (((tp - lo_t) ** 2 > d1) & ((hi_t - tp) ** 2 > d2))
        cnt = int(inside.sum())
        s[x] = k / max(cnt, 1) ** 0.25
        ri = rs_all[i0:i1][inside]
        dmin[x] = min(rad_all[a], rad_all[x],
                      float(ri.min()) if len(ri) else np.inf)
    return s, np.isfinite(s), dmin


def main():
    ap = argparse.ArgumentParser(description='Игра: линейка-отношение')
    ap.add_argument('--n', type=int, default=12000)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--radell', type=float, default=2.2)
    ap.add_argument('--dens', type=int, default=300)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--bands', type=int, default=4)
    ap.add_argument('--step', type=float, default=1.0)
    ap.add_argument('--guard', type=float, default=1.0,
                    help='запас в ell: весь интервал не ближе rad+guard')
    ap.add_argument('--blk', type=int, default=1500)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    rad = a.radell * ell
    print('ПРАВИЛО: из x назад k звеньев по самой долгой цепи -> a;')
    print('         n = число событий в (a, x);   s = k / n^(1/4)')
    print('Выведено: в плоском мире n = (k/m4)^4, то есть БЕЗ плотности.')
    print()
    print(f'N = {a.n}, ell = {ell:.5f}, k = {a.k}, засевов {a.seeds}')
    print(f'разрежение {rad:.4f} = {a.radell} ell вокруг центра')
    print()

    # ---- ПРОВЕРКА 3: плотностная независимость, напрямую ----
    print('ПРОВЕРКА ЛИНЕЙКИ: одно и то же s при РАЗНОЙ плотности?')
    print('(в пустом ромбе, без вещества; если s поедет — всё напрасно)')
    print(f"{'N':>8}{'ell':>9}{'s среднее':>12}{'ст.ош':>9}"
          f"{'n среднее':>11}{'ожидание (k/m4)^4':>19}")
    base = {}
    for NN in (4000, 8000, 16000):
        vals = []
        for s_ in range(3):
            X, isobs, _ = build(NN, a.t, np.random.default_rng(5000 + 61 * s_),
                                0.0, 4)
            Xb = X[~isobs]
            sv, ok, _ = ruler(Xb, a.k, a.blk)
            vals.append(np.nanmean(sv[ok]))
        v = np.array(vals)
        nmean = (a.k / v.mean()) ** 4
        base[NN] = v.mean()
        print(f'{NN:>8}{(math.pi/24.0/NN)**0.25:>9.5f}{v.mean():>12.4f}'
              f'{v.std(ddof=1)/math.sqrt(len(v)):>9.4f}{nmean:>11.2f}'
              f'{"— то же число":>19}')
    sp = np.array(list(base.values()))
    rel = (sp.max() - sp.min()) / sp.mean()
    print(f'  разброс s при четырёхкратной смене плотности: {rel*100:.2f} %')
    print(f'  {"ЛИНЕЙКА ПЛОТНОСТНО-НЕЗАВИСИМА" if rel < 0.05 else "ЛИНЕЙКА ЕДЕТ — правило негодно"}')
    print()

    # ---- отклик у вещества ----
    bands = [(rad + q * a.step * ell, rad + (q + 1) * a.step * ell)
             for q in range(a.bands)]
    bc = np.array([0.5 * (lo + hi) for lo, hi in bands])
    print('ОТКЛИК ЛИНЕЙКИ У ВЕЩЕСТВА, по радарным полосам')
    print('судья: у рассыпанного контроля отклик обязан быть НУЛЁМ')
    print()
    acc = {'дырка': [[] for _ in bands], 'рассыпано': [[] for _ in bands],
           'нуль': [[] for _ in bands]}
    for s_ in range(a.seeds):
        g = lambda: np.random.default_rng(61000 + 137 * s_)
        X0, iso0, _ = build(a.n, a.t, g(), 0.0, a.dens)
        Xb0 = X0[~iso0]
        s0, ok0, dm0 = ruler(Xb0, a.k, a.blk)
        tau0, rho0, fr0 = radar_chain(Xb0, X0[iso0])
        for lbl, rr, sc in (('дырка', rad, False), ('рассыпано', rad, True),
                            ('нуль', 0.0, False)):
            X1, iso1, _ = build(a.n, a.t, g(), rr, a.dens, sc)
            Xb1 = X1[~iso1]
            s1, ok1, dm1 = ruler(Xb1, a.k, a.blk)
            tau1, rho1, fr1 = radar_chain(Xb1, X1[iso1])
            for q, (lo, hi) in enumerate(bands):
                # ЗАЩИТА ОТ КАСАНИЯ: весь локальный интервал обязан
                # лежать не ближе rad + guard*ell к центру вещества,
                # иначе мерится прямое перекрытие, а не отклик поля
                gd = rad + a.guard * ell
                m0 = (ok0 & fr0 & (rho0 >= lo * a.dens)
                      & (rho0 < hi * a.dens) & (dm0 > gd))
                m1 = (ok1 & fr1 & (rho1 >= lo * a.dens)
                      & (rho1 < hi * a.dens) & (dm1 > gd))
                if m0.sum() < 20 or m1.sum() < 20:
                    continue
                acc[lbl][q].append(float(np.nanmean(s1[m1])
                                         - np.nanmean(s0[m0])))
            del X1, Xb1, s1
        del X0, Xb0, s0
        print(f'  засев {s_} ({time.time()-t0:.0f} с)', flush=True)

    print()
    print(f"{'r в ell':>9}{'ДЫРКА':>12}{'сигм':>7}{'рассыпано':>12}{'сигм':>7}"
          f"{'нуль':>10}{'во сколько раз':>16}")
    rat = []
    for q in range(a.bands):
        vh = np.array(acc['дырка'][q]); vs = np.array(acc['рассыпано'][q])
        vz = np.array(acc['нуль'][q])
        if len(vh) < 3 or len(vs) < 3:
            continue
        seh = vh.std(ddof=1) / math.sqrt(len(vh))
        ses = vs.std(ddof=1) / math.sqrt(len(vs))
        r = abs(vh.mean()) / max(abs(vs.mean()), 1e-12)
        rat.append(r)
        print(f'{bc[q]/ell:>9.2f}{vh.mean():>+12.5f}'
              f'{abs(vh.mean())/max(seh,1e-12):>7.1f}'
              f'{vs.mean():>+12.5f}{abs(vs.mean())/max(ses,1e-12):>7.1f}'
              f'{vz.mean():>+10.5f}{r:>16.1f}')
    print()
    if rat:
        print(f'  сигнал превышает рассыпанный контроль в '
              f'{np.mean(rat):.1f} раз в среднем')
        print(f'  для сравнения: у меры |I|^(1/4) было 2.4 раза, '
              f'и по форме — чистый сдвиг плотности')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
