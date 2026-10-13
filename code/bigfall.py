#!/usr/bin/env python3
"""ЗАКОН СПАДАНИЯ при N = 60000: ньютоновский предел из порядка.

ЧТО УЖЕ ЕСТЬ (`onepass-oct5.md`, §21). В радарном кадре измерен закон по
МАССЕ при фиксированной геометрии:

    M = 100 / 200 / 294 / 395  ->  дефицит хода часов
    -0.0066 / -0.0114 / -0.0163 / -0.0237   (4.7σ / 6.2σ / 8.1σ / 12.7σ)
    дефицит ~ M^+0.918

то есть линейно по массе, как велит слабое поле ОТО. Знак верен (часы у
вещества медленнее), рассыпанный контроль плоский, нулевая точка ровно
нуль.

Не хватает зависимости от РАССТОЯНИЯ. Причина считается: при N = 2200
радиус мира 5.7 ell, дырка 2.3 ell, чистого места под профиль полторы
ячейки. Нужно >= 10 ell снаружи объекта:

    радиус/ell = 0.5 * (7.64 N)^(1/4) >= 13   =>   N ≈ 60000

Вместе с линейностью по массе это даёт `дефицит ~ M/r` — ньютоновский
предел, выведенный из порядка.

ПОЧЕМУ ЭТО ВООБЩЕ СЧИТАЕТСЯ. В лоб N = 60000 не берётся: матрица
отношений есть 3.6e9 записей (3.6 ГБ в булях), а умножение для мощностей
интервалов 2.2e14 операций. Но полная матрица НЕ НУЖНА:

  высота            h[i] = 1 + max{h[j] : j < i} — просмотр по времени
                    КУСКАМИ, без хранения матрицы, O(N²) сравнений;
  радарные коорд.   для каждого события последнее событие цепи в прошлом
                    и первое в будущем — O(N * длина цепи) = 1.8e7, даром.

Мощности интервалов (самое дорогое) здесь не нужны вовсе: ход часов есть
высота на радарное время.

  python3 bigfall.py --n 60000 --seeds 1      # замер цены
  python3 bigfall.py --n 60000 --seeds 8
"""
import argparse
import math
import time

import numpy as np

SCRATCH = ("/tmp/claude-0/-home-claude/8f20417e-eecf-5054-8071-09c440210416/scratchpad")


def diamond(N, T, rs):
    out, got = [], 0
    while got < N:
        m = min(2_000_000, (N - got) * 10 + 4096)
        Y = np.empty((m, 4), np.float32)
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= r) & (T - Y[:, 0] >= r)]
        if len(Y):
            out.append(Y)
            got += len(Y)
    X = np.vstack(out)[:N]
    return X[np.argsort(X[:, 0])]


def heights_chunked(X, chunk=700):
    """h[i] = 1 + max{h[j] : j раньше i}. Куски, без матрицы N*N.

    События уже упорядочены по времени, поэтому прошлое i лежит в j < i.
    Сравнение КВАДРАТОВ, без корня: точнее и быстрее.
    """
    n = len(X)
    t = X[:, 0].astype(np.float32)
    S = X[:, 1:].astype(np.float32)
    g = (S * S).sum(1)
    h = np.ones(n, np.int32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        if a == 0:
            continue
        # кандидаты в прошлое: все j < b (в т.ч. внутри куска)
        Sj = S[:b]
        gj = g[:b]
        tj = t[:b]
        for i in range(a, b):
            dt = t[i] - tj[:i]
            if len(dt) == 0:
                continue
            d2 = g[i] + gj[:i] - 2.0 * (Sj[:i] @ S[i])
            np.maximum(d2, 0.0, out=d2)
            m = (dt > 0) & (dt * dt > d2)
            if m.any():
                h[i] = h[:i][m].max() + 1
    return h


def heights_blocked(X, blk=1500):
    """То же, но блоками: быстрее при больших N.

    Внутри блока зависимость последовательная, поэтому блок обрабатывается
    так: сначала вклад ВСЕХ предыдущих блоков разом (матрично), потом
    внутренняя последовательная досборка по блоку.
    """
    n = len(X)
    t = X[:, 0].astype(np.float32)
    S = X[:, 1:].astype(np.float32)
    g = (S * S).sum(1)
    h = np.ones(n, np.int32)
    for a in range(0, n, blk):
        b = min(a + blk, n)
        if a > 0:
            # вклад прошлых блоков: максимум h[j] по j < a, j < i
            dt = t[a:b][:, None] - t[None, :a]
            d2 = (g[a:b][:, None] + g[None, :a]
                  - 2.0 * (S[a:b] @ S[:a].T))
            np.maximum(d2, 0.0, out=d2)
            rel = (dt > 0) & (dt * dt > d2)
            hv = h[:a].astype(np.int32)
            best = np.where(rel, hv[None, :], 0).max(1)
            h[a:b] = np.maximum(h[a:b], best + 1)
            del dt, d2, rel
        # внутри блока — последовательно
        for i in range(a, b):
            if i == a:
                continue
            dt = t[i] - t[a:i]
            d2 = g[i] + g[a:i] - 2.0 * (S[a:i] @ S[i])
            np.maximum(d2, 0.0, out=d2)
            m = (dt > 0) & (dt * dt > d2)
            if m.any():
                h[i] = max(int(h[i]), int(h[a:i][m].max()) + 1)
    return h


def radar_chain(X, C):
    """Радарные координаты всех событий относительно цепи C. O(N*len(C))."""
    n, k = len(X), len(C)
    tC = C[:, 0].astype(np.float32)
    SC = C[:, 1:].astype(np.float32)
    gC = (SC * SC).sum(1)
    t = X[:, 0].astype(np.float32)
    S = X[:, 1:].astype(np.float32)
    g = (S * S).sum(1)
    ip = np.full(n, -1, np.int32)
    ifu = np.full(n, -1, np.int32)
    for a in range(0, n, 4000):
        b = min(a + 4000, n)
        dt = t[a:b][:, None] - tC[None, :]
        d2 = (g[a:b][:, None] + gC[None, :]
              - 2.0 * (S[a:b] @ SC.T))
        np.maximum(d2, 0.0, out=d2)
        past = (dt > 0) & (dt * dt > d2)
        fut = (dt < 0) & (dt * dt > d2)
        idx = np.arange(k)[None, :]
        pp = np.where(past, idx, -1).max(1)
        ff = np.where(fut, idx, k).min(1)
        ip[a:b] = pp
        ifu[a:b] = np.where(ff < k, ff, -1)
        del dt, d2, past, fut
    ok = (ip >= 0) & (ifu >= 0)
    tau = np.where(ok, 0.5 * (ip + ifu), np.nan)
    rho = np.where(ok, 0.5 * (ifu - ip), np.nan)
    return tau, rho, ok


def build(N, T, rs, rad, dens, scatter=False):
    bg = diamond(N, T, rs)
    nrem = 0
    if rad > 0:
        rr = np.linalg.norm(bg[:, 1:], axis=1)
        if scatter:
            k = int((rr < rad).sum())
            d = rs.choice(len(bg), k, replace=False)
            keep = np.ones(len(bg), bool)
            keep[d] = False
            nrem = k
        else:
            keep = rr >= rad
            nrem = int((~keep).sum())
        bg = bg[keep]
    Z = np.zeros((dens, 4), np.float32)
    Z[:, 0] = np.linspace(0.03, T - 0.03, dens)
    X = np.vstack([bg, Z])
    X = X[np.argsort(X[:, 0])]
    isobs = np.all(np.abs(X[:, 1:]) < 1e-9, axis=1)
    return X, isobs, nrem


def one(N, T, rs, rad, dens, scatter=False, blk=1500):
    """ПОЙМАННАЯ ОШИБКА, касавшаяся всех прежних замеров хода часов.

    Сначала высота считалась по ВСЕМ событиям, включая цепь наблюдателя. Но
    плотная цепь из `dens` событий сама есть цепь длины `dens`, тогда как
    естественная самая длинная цепь сприклинга идёт как N^0.25 (десяток-два).
    Значит высота события оказывалась почти равна его НОМЕРУ ВДОЛЬ ЦЕПИ, и
    отношение `h / tau` вырождалось по построению: при N = 8000 и цепи 300
    «высота мира» вышла ровно 300, и ни одна полоса не дала двух сигм.

    Починка разделяет роли: ПРИБОР даёт координаты, СИСТЕМА даёт собственное
    время. Высота считается ТОЛЬКО ПО ФОНУ, цепь из неё исключена.
    """
    X, isobs, nrem = build(N, T, rs, rad, dens, scatter)
    bi = np.flatnonzero(~isobs)
    Xb = X[bi]
    h = heights_blocked(Xb, blk)          # высота ТОЛЬКО по фону
    C = X[isobs]
    tau, rho, ok = radar_chain(Xb, C)
    return dict(tau=tau, rho=rho, ok=ok, h=h.astype(np.float64),
                nrem=nrem, cov=float(ok.mean()), hmax=int(h.max()))


def main():
    ap = argparse.ArgumentParser(description='Закон спадания при большом N')
    ap.add_argument('--n', type=int, default=60000)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--rad', type=float, default=0.077)
    ap.add_argument('--dens', type=int, default=300)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--bands', type=int, default=8)
    ap.add_argument('--blk', type=int, default=1500)
    ap.add_argument('--scatter', action='store_true')
    ap.add_argument('--seed0', type=int, default=0,
                    help='сдвиг номера засева: прогоны детерминированы, '
                         'и без сдвига повторный запуск даёт ТЕ ЖЕ миры')
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    print(f'N = {a.n}, ell = {ell:.5f}, радиус мира 0.5 = {0.5/ell:.1f} ell')
    print(f'дырка {a.rad} = {a.rad/ell:.1f} ell, цепь {a.dens} событий '
          f'(шаг {1.0/a.dens/ell:.2f} ell)')
    print(f'чистого места снаружи: {(0.5-a.rad)/ell:.1f} ell, '
          f'полос по 1 ell: {a.bands}')
    print(f'засевов {a.seeds}')
    print()
    print('ход часов = высота / радарное время; матрица N*N НЕ строится')
    print('ожидание: дефицит ~ 1/r (слабое поле ОТО) при линейности по M')
    print()

    # ЗАЩИТА ОТ ГРАНИЦЫ, поставленная после пойманной ошибки
    # воспроизводимости. Полосы обязаны лежать внутри мира: радиус ромба
    # есть T/2, но у краёв по времени он сужается, поэтому берётся
    # запас — не дальше 0.80 от максимального радиуса. Без этой защиты
    # команда с числом полос по умолчанию при N = 8000 уходила на 0.586
    # при радиусе 0.5, две внешние полосы были границей, и показатель
    # выходил -1.238 вместо -1.04.
    rmax_safe = 0.80 * (a.t / 2.0)
    nb_fit = int(max(1, math.floor((rmax_safe - a.rad) / ell)))
    nb = min(a.bands, nb_fit)
    if nb < a.bands:
        print(f'  полос урезано {a.bands} -> {nb}: внешний край не далее '
              f'{rmax_safe:.3f} ({rmax_safe/ell:.1f} ell) от центра')
    bands = [(a.rad + k * ell, a.rad + (k + 1) * ell)
             for k in range(nb)]
    acc = [[] for _ in bands]
    accS = [[] for _ in bands]
    nr, cov, hm = [], [], []
    for s in range(a.seed0, a.seed0 + a.seeds):
        ts = time.time()
        A = one(a.n, a.t, np.random.default_rng(90000 + 181 * s), 0.0,
                a.dens, blk=a.blk)
        B = one(a.n, a.t, np.random.default_rng(90000 + 181 * s), a.rad,
                a.dens, blk=a.blk)
        S = (one(a.n, a.t, np.random.default_rng(90000 + 181 * s), a.rad,
                 a.dens, scatter=True, blk=a.blk) if a.scatter else None)
        nr.append(B['nrem']); cov.append(B['cov']); hm.append(B['hmax'])
        for q, (lo, hi) in enumerate(bands):
            lo_d, hi_d = lo * a.dens, hi * a.dens
            mA = A['ok'] & (A['rho'] >= lo_d) & (A['rho'] < hi_d)
            mB = B['ok'] & (B['rho'] >= lo_d) & (B['rho'] < hi_d)
            if mA.sum() < 20 or mB.sum() < 20:
                continue
            rA = A['h'][mA] / np.maximum(A['tau'][mA], 1e-9)
            rB = B['h'][mB] / np.maximum(B['tau'][mB], 1e-9)
            acc[q].append(float(rB.mean() - rA.mean()))
            if S is not None:
                mS = S['ok'] & (S['rho'] >= lo_d) & (S['rho'] < hi_d)
                if mS.sum() >= 20:
                    rS = S['h'][mS] / np.maximum(S['tau'][mS], 1e-9)
                    accS[q].append(float(rS.mean() - rA.mean()))
        del A, B, S
        print(f'  засев {s}: {time.time()-ts:.0f} с, удалено {nr[-1]}, '
              f'покрытие {cov[-1]:.3f}, высота мира {hm[-1]}', flush=True)

    print()
    print(f"{'r от центра':>12}{'в ell':>7}{'дефицит':>12}{'ст.ош':>10}"
          f"{'сигм':>7}{'миров':>7}" +
          (f"{'рассыпано':>12}{'сигм':>7}" if a.scatter else ''))
    xs, ys = [], []
    for q, (lo, hi) in enumerate(bands):
        v = np.array(acc[q])
        if len(v) < 3:
            continue
        se = v.std(ddof=1) / math.sqrt(len(v))
        c = 0.5 * (lo + hi)
        line = (f'{c:>12.4f}{c/ell:>7.1f}{v.mean():>+12.6f}{se:>10.6f}'
                f'{abs(v.mean())/max(se,1e-12):>7.1f}{len(v):>7}')
        if a.scatter and len(accS[q]) >= 3:
            w = np.array(accS[q])
            sw = w.std(ddof=1) / math.sqrt(len(w))
            line += (f'{w.mean():>+12.6f}'
                     f'{abs(w.mean())/max(sw,1e-12):>7.1f}')
        print(line)
        if v.mean() < 0 and abs(v.mean()) / max(se, 1e-12) > 2:
            xs.append(c); ys.append(-v.mean())
    if len(xs) >= 4:
        b, c0 = np.polyfit(np.log(xs), np.log(ys), 1)
        pred = b * np.log(xs) + c0
        ss = ((np.log(ys) - np.mean(np.log(ys))) ** 2).sum()
        r2 = 1 - ((np.log(ys) - pred) ** 2).sum() / max(ss, 1e-18)
        print()
        print(f'  СПАДАНИЕ ~ r^{b:+.3f}   R² = {r2:.4f}   точек {len(xs)}')
        print(f'  слабое поле ОТО: -1')
    else:
        print(f'\n  точек выше 2 сигм: {len(xs)} — для показателя мало')

    # ---- ДОПИСАНО 8 октября: складной нож ПО ЗАСЕВАМ для показателя ----
    # Зачем. Две оценки ошибки показателя расходились втрое: разброс
    # вокруг прямой (±0.013) против распространения поячеечных ошибок
    # как НЕЗАВИСИМЫХ (±0.10). Но полосы считаются по одним и тем же
    # миpам, значит статистическая ошибка сдвигает их ВМЕСТЕ и на
    # показатель почти не влияет. Складной нож по засевам учитывает обе
    # стороны сразу и не требует знать ковариацию полос.
    import pickle as _pk
    cent = [(a.rad + (k + 0.5) * ell) / ell for k in range(nb)]
    per = [list(acc[q]) for q in range(nb)]
    ns = min(len(p) for p in per) if per else 0
    try:
        with open(f'{SCRATCH}/bigfall_{a.n}_{a.seed0}.pkl', 'wb') as fh:
            _pk.dump({'r_ell': cent, 'per_seed': per, 'n': a.n,
                      'rad': a.rad, 'ell': ell}, fh)
        print(f'  поячеечные значения по засевам сохранены '
              f'({nb} полос x {ns} засевов)')
    except Exception as exc:
        print(f'  не удалось сохранить: {exc}')

    def _slope(vals, sl):
        x = np.log(np.array(cent)[sl])
        y = np.array(vals)[sl]
        m = y < 0
        if m.sum() < 3:
            return float('nan')
        return float(np.polyfit(x[m], np.log(-y[m]), 1)[0])

    if ns >= 3 and nb >= 6:
        A = np.array([[per[q][s] for q in range(nb)] for s in range(ns)])
        half = nb // 2
        wins = (('все', slice(None)), ('внутр.', slice(0, half + 1)),
                ('внешн.', slice(half, None)))
        print()
        print('  СКЛАДНОЙ НОЖ ПО ЗАСЕВАМ (единица независимости — мир)')
        print(f"  {'окно':>10}{'показатель':>14}{'±':>9}")
        res = {}
        for lab, sl in wins:
            full = _slope(A.mean(axis=0), sl)
            jk = np.array([_slope(np.delete(A, i, axis=0).mean(axis=0), sl)
                           for i in range(ns)])
            jk = jk[np.isfinite(jk)]
            se = (math.sqrt((len(jk) - 1) / len(jk)
                            * ((jk - jk.mean()) ** 2).sum())
                  if len(jk) > 1 else float('nan'))
            print(f'  {lab:>10}{full:>14.3f}{se:>9.3f}')
            res[lab] = (full, se, jk)
        if 'внутр.' in res and 'внешн.' in res:
            d = res['внутр.'][0] - res['внешн.'][0]
            dj = res['внутр.'][2] - res['внешн.'][2]
            sd = (math.sqrt((len(dj) - 1) / len(dj)
                            * ((dj - dj.mean()) ** 2).sum())
                  if len(dj) > 1 else float('nan'))
            print(f'  РАЗНОСТЬ внутр. − внешн. = {d:+.3f} ± {sd:.3f}'
                  f'  = {d/max(sd,1e-9):+.1f} сигм')
            print('  (ноль ⟹ степенной закон един; не ноль ⟹ дрейф реален)')

    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
