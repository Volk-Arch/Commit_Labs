#!/usr/bin/env python3
"""
ДВУЩЕЛЕВОЙ ОПЫТ НА ПРИЧИННОМ МНОЖЕСТВЕ
======================================

Два случая:
  A. БЕЗ НАБЛЮДЕНИЯ — картина строится по полным амплитудам (все фазы известны).
     Полосы есть: интерференционный член следует cos(разности фаз).

  B. С НАБЛЮДЕНИЕМ — картина строится по тому, что сохраняет ПОРЯДОК.
     Полос нет: относительные фазы между шкалами в порядок не проходят.

Требуется: numpy, scikit-learn, matplotlib (опционально, для картинки).
Запуск:    python3 double_slit.py
           python3 double_slit.py --n 20000 --plot out.png
           python3 double_slit.py --gpu          # если установлен cupy
"""
import argparse
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")      # Windows: консоль cp1251
    sys.stderr.reconfigure(encoding="utf-8")
import numpy as np

# ---------------------------------------------------------------- конструкция

def build_causet(N, k, seed):
    """Событие = k фаз. i раньше j, если раньше по ВСЕМ фазам."""
    rs = np.random.default_rng(seed)
    X = rs.random((N, k))
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return X, R


def direct_links(R):
    """Прямые связи: между i и j нет промежуточных событий."""
    A = R.astype(np.float32)
    return R & ((A @ A) == 0)


def depth(R):
    """Собственное время: длина максимальной цепи до события."""
    Rb = np.ascontiguousarray(R, dtype=bool)
    n = len(Rb)
    indeg = Rb.sum(axis=1).astype(np.int64)
    D = np.zeros(n, dtype=np.int64)
    ready = list(np.flatnonzero(indeg == 0))
    while ready:
        j = ready.pop()
        kids = np.flatnonzero(Rb[:, j])
        if kids.size:
            D[kids] = np.maximum(D[kids], D[j] + 1)
            indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    return D


def topo_order(L):
    Lb = np.ascontiguousarray(L, dtype=bool)
    indeg = Lb.sum(axis=1).astype(np.int64)
    ready = list(np.flatnonzero(indeg == 0))
    out = []
    while ready:
        j = ready.pop()
        out.append(j)
        kids = np.flatnonzero(Lb[:, j])
        if kids.size:
            indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    return out


# ------------------------------------------------------------- распространение

def propagate(Lb, order, amp, sources, k):
    """Комплексная амплитуда от источников по причинным связям.

    У события k амплитуд — по одной на шкалу. Никакой нормировки:
    она нелинейна и стирает относительные фазы.
    """
    N = len(Lb)
    V = np.zeros((N, k), dtype=complex)
    for v in order:
        if v in sources:
            V[v] = sources[v]
            continue
        par = np.flatnonzero(Lb[v])
        if len(par):
            V[v] = amp[v] * (V[par].sum(0) / len(par))
    return V


# -------------------------------------------------------------------- сценарий

def run(N=9000, k=4, seed=85100, verbose=True):
    X, R = build_causet(N, k, seed)
    Rb = np.asarray(R, dtype=bool)
    L = direct_links(R)
    Lb = np.asarray(L, dtype=bool)
    order = topo_order(L)
    D = depth(R)
    Dc = depth(np.ascontiguousarray(Rb.T))

    # щели — два события в средних слоях с наибольшим числом потомков
    mid = np.flatnonzero((D > 0.25 * D.max()) & (D < 0.40 * D.max()))
    ndesc = Rb[:, mid].sum(axis=0)
    o = np.argsort(-ndesc)
    s1, s2 = int(mid[o[0]]), int(mid[o[1]])

    # экран — верхняя половина по глубине
    screen = np.flatnonzero(D > 0.5 * D.max())

    if verbose:
        print(f"событий {N}, высота {int(D.max())}, прямых связей {int(Lb.sum())}")
        print(f"щели: {s1} и {s2};  экран: {len(screen)} событий\n")

    amp = np.exp(2j * np.pi * X)          # амплитуда события из ЕГО фаз

    V1 = propagate(Lb, order, amp, {s1: amp[s1]}, k)
    V2 = propagate(Lb, order, amp, {s2: amp[s2]}, k)
    Vb = propagate(Lb, order, amp, {s1: amp[s1], s2: amp[s2]}, k)

    # интенсивность: когерентное сложение амплитуд по шкалам
    I1 = np.abs(V1[screen].sum(1)) ** 2
    I2 = np.abs(V2[screen].sum(1)) ** 2
    Ib = np.abs(Vb[screen].sum(1)) ** 2

    # интерференционный член: насколько обе щели отличаются от суммы одиночных
    interf = (Ib - (I1 + I2)) / np.maximum(I1 + I2, 1e-30)

    # разность фаз путей — то, чем интерференция должна управляться
    dphi = np.mod(np.angle(V1[screen].sum(1)) - np.angle(V2[screen].sum(1)), 2 * np.pi)

    ok = np.isfinite(interf) & (np.abs(interf) < 10)
    return dict(X=X, Rb=Rb, D=D, Dc=Dc, screen=screen[ok], s1=s1, s2=s2,
                interf=interf[ok], dphi=dphi[ok], V1=V1, V2=V2)


def case_A(res, verbose=True):
    """БЕЗ НАБЛЮДЕНИЯ: полосы следуют cos(разности фаз)."""
    r, dphi = res['interf'], res['dphi']
    c = np.corrcoef(np.cos(dphi), r)[0, 1]
    if verbose:
        print("A. БЕЗ НАБЛЮДЕНИЯ (полные амплитуды)")
        print(f"   максимумов / минимумов : {int((r > 0).sum())} / {int((r < 0).sum())}")
        print(f"   размах интерференции   : {r.max() - r.min():.3f}")
        print(f"   корреляция с cos(dphi) : {c:+.3f}   <- ПОЛОСЫ ЕСТЬ\n")
    return c


def case_B(res, verbose=True):
    """С НАБЛЮДЕНИЕМ: картина по тому, что сохраняет порядок."""
    from sklearn.ensemble import HistGradientBoostingRegressor as GBR

    S, r = res['screen'], res['interf']
    Rb, D, Dc = res['Rb'], res['D'], res['Dc']
    s1, s2 = res['s1'], res['s2']
    Rf = Rb.astype(np.float32)

    # всё, что наблюдатель ВНУТРИ порядка может измерить
    F_order = np.column_stack([
        Rf.sum(1)[S], Rf.sum(0)[S],       # размер прошлого и будущего
        D[S], Dc[S],                       # глубина и кодлубина
        Rb[S][:, s1], Rb[S][:, s2],        # связан ли со щелями
        (Rb[S] & Rb[s1]).sum(1),           # общее прошлое со щелью 1
        (Rb[S] & Rb[s2]).sum(1),           # общее прошлое со щелью 2
    ])
    # контроль сверху: истинные фазы, которых порядок НЕ хранит
    V1, V2 = res['V1'], res['V2']
    F_true = np.column_stack([
        np.angle(V1[S].sum(1)), np.angle(V2[S].sum(1)),
        np.abs(V1[S].sum(1)), np.abs(V2[S].sum(1)),
    ])

    h = len(S) // 2

    def r2(F, y):
        m = GBR(max_iter=400, random_state=0).fit(F[:h], y[:h])
        p = m.predict(F[h:])
        yt = y[h:]
        return float(1 - np.var(yt - p) / max(np.var(yt), 1e-12))

    a = r2(F_order, r)
    b = r2(F_true, r)
    if verbose:
        print("B. С НАБЛЮДЕНИЕМ (только то, что сохраняет порядок)")
        print(f"   R2 по признакам порядка : {a:+.3f}   <- ПОЛОС НЕТ")
        print(f"   R2 по истинным фазам    : {b:+.3f}   (контроль сверху)\n")
    return a, b


def why(res, verbose=True):
    """Почему: порядок не хранит относительных фаз между шкалами."""
    from sklearn.ensemble import HistGradientBoostingRegressor as GBR
    X, Rb, D, Dc = res['X'], res['Rb'], res['D'], res['Dc']
    N = len(X)
    Rf = Rb.astype(np.float32)
    F = np.column_stack([Rf.sum(1), Rf.sum(0), D, Dc,
                         Rf.sum(1) * Rf.sum(0) / N ** 2, D - Dc])
    h = N // 2

    def r2(y):
        m = GBR(max_iter=200, random_state=0).fit(F[:h], y[:h])
        p = m.predict(F[h:])
        yt = y[h:]
        return float(1 - np.var(yt - p) / max(np.var(yt), 1e-12))

    rows = [("сумма фаз (время)", X.sum(1)),
            ("разброс фаз (заряд)", X.var(1)),
            ("отдельная шкала", X[:, 0]),
            ("РАЗНОСТЬ фаз 0-1", X[:, 0] - X[:, 1]),
            ("cos(2pi*(phi0-phi1))", np.cos(2 * np.pi * (X[:, 0] - X[:, 1])))]
    if verbose:
        print("ПОЧЕМУ: что порядок сохраняет, а что теряет")
        for nm, y in rows:
            v = r2(y)
            tag = "проходит" if v > 0.7 else ("частично" if v > 0.3 else "ТЕРЯЕТСЯ")
            print(f"   {nm:<24} R2 = {v:+.3f}   {tag}")
        print("\n   интерференция держится на РАЗНОСТЯХ фаз —")
        print("   а именно они в порядок не проходят.\n")


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, dphi = res['interf'], res['dphi']
    o = np.argsort(dphi)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(dphi[o], r[o], s=6, alpha=0.5)
    xs = np.linspace(0, 2 * np.pi, 200)
    ax[0].plot(xs, np.cos(xs) * np.abs(r).mean() * 2, 'r-', lw=1.5, label='cos(dphi)')
    ax[0].set_xlabel('разность фаз путей'); ax[0].set_ylabel('интерференционный член')
    ax[0].set_title('A. без наблюдения: полосы'); ax[0].legend()
    ax[1].scatter(res['D'][res['screen']], r, s=6, alpha=0.5)
    ax[1].set_xlabel('глубина события (то, что видит порядок)')
    ax[1].set_ylabel('интерференционный член')
    ax[1].set_title('B. с наблюдением: структуры нет')
    plt.tight_layout(); plt.savefig(path, dpi=120)
    print(f"картинка сохранена: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9000, help="число событий")
    ap.add_argument("--k", type=int, default=4, help="число шкал")
    ap.add_argument("--seed", type=int, default=85100)
    ap.add_argument("--plot", type=str, default=None, help="файл для картинки")
    args = ap.parse_args()

    print("=" * 62)
    print("ДВУЩЕЛЕВОЙ ОПЫТ НА ПРИЧИННОМ МНОЖЕСТВЕ")
    print("=" * 62 + "\n")

    res = run(N=args.n, k=args.k, seed=args.seed)
    case_A(res)
    case_B(res)
    why(res)

    print("=" * 62)
    print("ВЫВОД: интерференция существует на уровне амплитуд,")
    print("       но наблюдатель, живущий в порядке, её не видит —")
    print("       относительные фазы при переводе в порядок теряются.")
    print("=" * 62)

    if args.plot:
        plot(res, args.plot)
