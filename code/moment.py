#!/usr/bin/env python3
"""ВСТРЕЧИ В МОМЕНТЕ: сохраняются ли соседи от шага к шагу.

ЕГО ПОПРАВКА, и она бьёт точно в мою ошибку. `tallmeet.py` копил встречи
ПО ВСЕЙ ИСТОРИИ — сорок два шага высоты, — поэтому граф вышел полным
(степень 113.9 из 116). Пространство есть СРЕЗ, а не история; складывать
надо в моменте: «вычистить граф от пересечений, оставить только порядок,
а встречи считать в моменте».

ЧТО ЭТО МЕНЯЕТ В ПОСТАНОВКЕ. Прежние проверки спрашивали про РОСТ ШАРА,
а он требует локальности — её нет: `|Δx|` связи ≈ 0.43 при радиусе мира
1.0 и не зависит от N (`locality.py`), так что диаметр любого графа
встреч ~2 при любом измельчении. «В моменте» позволяет спросить другое и
более глубокое:

    СОХРАНЯЮТСЯ ЛИ СОСЕДИ ОТ МОМЕНТА К МОМЕНТУ

Если у истории сейчас одни соседи, а через шаг совсем другие, то никакой
устойчивой структуры нет ни в каком смысле. Если те же — структура есть,
и плотность графа этому не мешает. Устойчивость не требует локальности,
поэтому эта проверка не обязана умереть от длинных связей.

УСТРОЙСТВО. Порядок чистится от пересечений: оставляем только
внутриисторийные отношения, то есть набор параллельных цепей — чистое
время. Встречи возвращаются отдельной структурой, по моментам: история A
встретила B в момент h, если есть ПРЯМАЯ СВЯЗЬ, нижний конец которой
стоит на высоте h, между событием A и событием B. (Внутри одного слоя
связей быть не может: равная высота означает несравнимость.)

МЕРИМ:
  перекрытие соседей (Жаккар) между моментами h и h+1;
  то же через 2, 3, 5 шагов — должно спадать, если структура реальна;
  КОНТРОЛЬ: те же связи, перемешанные по моментам — та же плотность, но
    устойчивости быть не должно. Это главный контроль;
  отдельно: ближе ли соседи в моменте, чем случайная пара (координаты
    только для проверки, в построении их нет).

  python3 moment.py
  python3 moment.py --n 3000 --t 6 --r 0.45
"""
import argparse
import math
import time

import numpy as np

from tallmeet import tall, greedy_long


def jac(a, b):
    if not a and not b:
        return float('nan')
    u = len(a | b)
    return len(a & b) / u if u else float('nan')


def per_moment(owner, hgt, li, lj, nh, hmax, shuffle_rs=None):
    """Соседи каждой истории по моментам. Момент = высота нижнего конца."""
    mom = hgt[li]
    if shuffle_rs is not None:
        mom = shuffle_rs.permutation(mom)
    nb = [dict() for _ in range(hmax + 2)]
    for t in range(len(li)):
        u, v = int(owner[li[t]]), int(owner[lj[t]])
        if u == v or u < 0 or v < 0:
            continue
        m = int(mom[t])
        if m > hmax:
            continue
        nb[m].setdefault(u, set()).add(v)
        nb[m].setdefault(v, set()).add(u)
    return nb


def persistence(nb, lags):
    """Перекрытие соседей одной истории между моментами, по сдвигам."""
    out = {L: [] for L in lags}
    H = len(nb)
    for L in lags:
        for m in range(H - L):
            a, b = nb[m], nb[m + L]
            for u in a:
                if u in b:
                    j = jac(a[u], b[u])
                    if np.isfinite(j):
                        out[L].append(j)
    return {L: (float(np.mean(v)) if v else float('nan'), len(v))
            for L, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description='Встречи в моменте')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--t', type=float, default=6.0)
    ap.add_argument('--r', type=float, default=0.45)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    vol = a.t * (4.0 * math.pi / 3.0) * a.r ** 3
    ell = (vol / a.n) ** 0.25
    print(f'N = {a.n}, цилиндр T = {a.t}, R = {a.r}; ell = {ell:.4f}')
    print(f'высота {a.t/ell:.1f} длин дискретности, ширина {a.r/ell:.1f}')
    print('встреча в момент h: прямая связь с нижним концом на высоте h')
    print('ГЛАВНЫЙ КОНТРОЛЬ: те же связи, перемешанные по моментам')
    print()
    lags = [1, 2, 3, 5]
    print(f"{'засев':>6}{'историй':>9}{'момент.':>9}{'степень':>9}"
          f"{'':>3}" + ''.join(f'{f"Ж(+{L})":>10}' for L in lags) +
          f"{'':>3}" + ''.join(f'{f"контр+{L}":>10}' for L in lags))

    rows, ctrl = [], []
    for s in range(a.seeds):
        rs = np.random.default_rng(9700 + 59 * s)
        X = tall(a.n, a.t, a.r, rs)
        dt = X[:, 0][None, :] - X[:, 0][:, None]
        dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=2)
        R = (dt > 0) & (dt > dx)
        del dt, dx
        A = R.astype(np.float32)
        LK = R & ~((A @ A) > 0.5)
        del A
        h = np.ones(a.n, np.int64)
        for i in range(a.n):
            pre = np.flatnonzero(R[:, i])
            if len(pre):
                h[i] = h[pre].max() + 1
        seqs, owner = greedy_long(R)
        nh = len(seqs)
        li, lj = np.nonzero(LK)
        hmax = int(h.max())

        nb = per_moment(owner, h, li, lj, nh, hmax)
        nbc = per_moment(owner, h, li, lj, nh, hmax,
                         shuffle_rs=np.random.default_rng(3 + s))
        p = persistence(nb, lags)
        pc = persistence(nbc, lags)
        deg = np.mean([len(v) for m in nb for v in m.values()] or [0])
        nm = sum(1 for m in nb if m)
        rows.append([p[L][0] for L in lags])
        ctrl.append([pc[L][0] for L in lags])
        print(f'{s:>6}{nh:>9}{nm:>9}{deg:>9.1f}{"":>3}' +
              ''.join(f'{p[L][0]:>10.4f}' for L in lags) + f'{"":>3}' +
              ''.join(f'{pc[L][0]:>10.4f}' for L in lags), flush=True)

        if s == 0:
            cen = np.array([X[q][:, 1:].mean(0) for q in seqs])
            near, rand = [], []
            rs2 = np.random.default_rng(77)
            for m in nb:
                for u, vs in m.items():
                    for v in vs:
                        near.append(float(np.linalg.norm(cen[v] - cen[u])))
                    for _ in range(len(vs)):
                        w = int(rs2.integers(0, nh))
                        rand.append(float(np.linalg.norm(cen[w] - cen[u])))
            print(f'    соседи в моменте: истинное расстояние '
                  f'{np.mean(near):.4f} против случайной пары '
                  f'{np.mean(rand):.4f}  ({len(near)} пар)')
        del R, LK, X
    print()
    m = np.mean(rows, axis=0)
    c = np.mean(ctrl, axis=0)
    print('СВОДКА: перекрытие соседей (Жаккар)')
    print(f"{'сдвиг':>8}{'правда':>10}{'контроль':>11}{'отношение':>12}")
    for t, L in enumerate(lags):
        print(f'{L:>8}{m[t]:>10.4f}{c[t]:>11.4f}'
              f'{m[t]/max(c[t],1e-9):>12.2f}')
    print()
    print('  правда заметно выше контроля и спадает со сдвигом =>')
    print('  соседство УСТОЙЧИВО, структура есть')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
