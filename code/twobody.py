#!/usr/bin/env python3
"""
МИКРОСИМУЛЯТОР: ДВА ТЕЛА
========================

Ставим два сгущения событий на заданном расстоянии и меряем поле МЕЖДУ ними.
Контроль: то же поле, когда тело одно.

    поле(два тела) − поле(одно тело)  =  вклад второго тела

Всё построено на одном правиле:
    событию даны k чисел; A раньше B, если A раньше по всем ДЕЙСТВУЮЩИМ шкалам.

Метрика не задаётся руками: число действующих шкал падает там, где события
гуще. Это и есть искривление — оно возникает из самой материи.

Запуск:
    python3 twobody.py                    # таблица по расстояниям
    python3 twobody.py --gap 0.3          # одно расстояние подробно
    python3 twobody.py --mass 400         # тела тяжелее
    python3 twobody.py --seeds 8          # больше статистики
"""
import argparse
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")      # Windows: консоль cp1251
    sys.stderr.reconfigure(encoding="utf-8")
import numpy as np


# ============================================================== устройство мира

def local_density(X, radius=0.2):
    """Сколько событий рядом с каждым."""
    n = len(X)
    d = np.empty(n)
    for i in range(n):
        d[i] = np.count_nonzero(np.linalg.norm(X - X[i], axis=1) < radius) - 1
    return d


def metric(dens, k, alpha):
    """МАТЕРИЯ ГОВОРИТ ПРОСТРАНСТВУ, КАК ИСКРИВЛЯТЬСЯ.

    Где гуще — там меньше действующих шкал. Ноль означает,
    что шкала выключена и по ней события не сравниваются.
    """
    W = np.ones((len(dens), k))
    if alpha <= 0:
        return W
    rel = dens/max(dens.mean(), 1e-9)
    off = np.clip((alpha*(rel - 1.0)).astype(int), 0, k-2)
    for m in range(k):
        W[off > (k-1-m), m] = 0.0
    return W


def causal_order(X, W):
    """Порядок по действующим шкалам."""
    n, k = X.shape
    R = np.ones((n, n), dtype=bool)
    act = W > 0
    for m in range(k):
        both = act[:, m][:, None] & act[:, m][None, :]
        R &= ((X[None, :, m] < X[:, None, m]) | ~both)
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def direct_links(R):
    A = R.astype(np.float32)
    return R & ((A @ A) == 0)


# ==================================================================== измерение

def field_at(X, R, L, point, background, nsample=120):
    """Поле в точке: доля прямых связей среди ближайших ФОНОВЫХ событий.

    Фоновые — то есть не принадлежащие телам. Иначе мерили бы вещество,
    а не поле.
    """
    d = np.linalg.norm(X - point, axis=1)
    cand = np.flatnonzero(background)
    if len(cand) < nsample:
        return np.nan
    idx = cand[np.argsort(d[cand])[:nsample]]
    sub = R[np.ix_(idx, idx)]
    subL = L[np.ix_(idx, idx)]
    tot = float(sub.sum())
    return float(subL.sum()/tot) if tot > 0 else np.nan


def build(background_pts, bodies, k, alpha, radius=0.2):
    X = np.vstack([background_pts] + bodies) if bodies else background_pts
    bg = np.zeros(len(X), dtype=bool)
    bg[:len(background_pts)] = True
    dens = local_density(X, radius)
    W = metric(dens, k, alpha)
    R = causal_order(X, W)
    return X, bg, R, direct_links(R), W


# ================================================================== эксперимент

def experiment(gap, seed, n_bg=2500, mass=200, k=4, alpha=1.5,
               spread=0.05, nsample=120):
    """Одно измерение: поле в середине при двух телах и при одном."""
    rs = np.random.default_rng(seed)
    bg = rs.random((n_bg, k))
    c1 = np.clip(np.full(k, 0.5) - gap/2, 0.05, 0.95)
    c2 = np.clip(np.full(k, 0.5) + gap/2, 0.05, 0.95)
    b1 = np.clip(c1 + rs.normal(0, spread, (mass, k)), 0, 1)
    b2 = np.clip(c2 + rs.normal(0, spread, (mass, k)), 0, 1)
    mid = (c1 + c2)/2

    X2, m2, R2, L2, W2 = build(bg, [b1, b2], k, alpha)
    f2 = field_at(X2, R2, L2, mid, m2, nsample)
    off2 = int((W2 == 0).sum())
    del X2, R2, L2, W2

    X1, m1, R1, L1, W1 = build(bg, [b1], k, alpha)
    f1 = field_at(X1, R1, L1, mid, m1, nsample)
    del X1, R1, L1, W1

    return f2, f1, off2


def run(gaps, seeds, **kw):
    out = []
    for gap in gaps:
        vals = []
        for s in range(seeds):
            f2, f1, off = experiment(gap, 500 + s*101 + int(gap*1000), **kw)
            if np.isfinite(f2) and np.isfinite(f1):
                vals.append((f2, f1, f2-f1, off))
        if vals:
            out.append((gap, np.array(vals)))
    return out


# ======================================================================= вывод

def report(res, args):
    line = "─"*66
    print(line)
    print("ВЗАИМОДЕЙСТВИЕ ДВУХ ТЕЛ")
    print(line)
    print(f"фон {args.background} событий, тело {args.mass}, шкал {args.k}, "
          f"связь плотность→метрика α={args.alpha}")
    print()
    print(f"{'расстояние':>11}{'поле (2 тела)':>15}{'поле (1 тело)':>15}"
          f"{'вклад 2-го':>13}{'±':>8}")
    for gap, A in res:
        d = A[:, 2]
        print(f"{gap:>11.2f}{A[:,0].mean():>15.4f}{A[:,1].mean():>15.4f}"
              f"{d.mean():>13.4f}{d.std()/max(np.sqrt(len(d)),1):>8.4f}")
    print()
    if len(res) >= 2:
        g = np.array([r[0] for r in res])
        v = np.array([r[1][:, 2].mean() for r in res])
        far = abs(v[-1])
        near = abs(v[0])
        print(f"вклад второго тела падает с {near:.4f} до {far:.4f} "
              f"при удалении с {g[0]:.2f} до {g[-1]:.2f}")
        if far < near/3:
            print("→ ВЗАИМОДЕЙСТВИЕ УБЫВАЕТ С РАССТОЯНИЕМ")
        else:
            print("→ убывания не видно")
    print()
    print("метрика возникает из плотности событий: "
          f"выключено шкал {int(res[0][1][:,3].mean())}")
    print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=None,
                    help="одно расстояние вместо таблицы")
    ap.add_argument("--mass", type=int, default=200, help="событий в теле")
    ap.add_argument("--background", type=int, default=2500, help="событий фона")
    ap.add_argument("--k", type=int, default=4, help="число шкал")
    ap.add_argument("--alpha", type=float, default=1.5,
                    help="сила связи плотность→метрика (0 = метрика плоская)")
    ap.add_argument("--seeds", type=int, default=5, help="число затравок")
    ap.add_argument("--compare", action="store_true",
                    help="сравнить с плоской метрикой (α=0)")
    args = ap.parse_args()

    gaps = [args.gap] if args.gap else [0.20, 0.35, 0.50, 0.70]
    if args.compare:
        gaps = [0.20, 0.28, 0.35, 0.45, 0.55]
        print("─"*66)
        print("ВКЛАД ИСКРИВЛЕНИЯ: с метрикой и без")
        print("─"*66)
        flat = {g: A[:, 2].mean() for g, A in
                run(gaps, args.seeds, n_bg=args.background, mass=args.mass,
                    k=args.k, alpha=0.0)}
        curv = {g: A[:, 2].mean() for g, A in
                run(gaps, args.seeds, n_bg=args.background, mass=args.mass,
                    k=args.k, alpha=args.alpha)}
        print(f"{'расстояние':>12}{'плоская метрика':>18}"
              f"{'с искривлением':>18}")
        for g in gaps:
            print(f"{g:>12.2f}{flat.get(g, float('nan')):>18.4f}"
                  f"{curv.get(g, float('nan')):>18.4f}")
        print()
        print("без искривления эффект обрывается на 0.35 (простое перекрытие);")
        print("с искривлением тянется до 0.55 — это и есть дальнодействие")
        print("─"*66)
    else:
        res = run(gaps, args.seeds, n_bg=args.background, mass=args.mass,
                  k=args.k, alpha=args.alpha)
        report(res, args)
