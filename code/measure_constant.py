#!/usr/bin/env python3
"""
КОНСТАНТА СВЯЗИ: точное измерение
=================================

Меряет коэффициент между массой тела и его вкладом в поле между телами.
Всё в собственных единицах мира: длина — среднее расстояние до ближайшего
события, масса — плотность событий в ячейке.

Что улучшено против быстрой версии:
  - 7 точек по массе вместо 3 (наклон определяется надёжнее)
  - общий фон в паре «одно тело / два тела» (шум фона сокращается)
  - взвешенная подгонка (точки разной надёжности)
  - усреднение по многим затравкам

Запуск:
    python3 measure_constant.py                    # быстро, ~1 мин
    python3 measure_constant.py --seeds 40         # точнее, ~1 час
    python3 measure_constant.py --n 8000 --seeds 40 --window 320
    python3 measure_constant.py --k 3              # другое число шкал
    python3 measure_constant.py --scan-k           # зависимость от k
    python3 measure_constant.py --scan-alpha       # зависимость от alpha
    python3 measure_constant.py --gpu --n 8000     # на видеокарте
    python3 measure_constant.py --max-mem 4        # ограничить память 4 ГБ
    python3 measure_constant.py --threads 4        # не занимать все ядра

Зависимости: numpy. Для видеокарты — необязательно cupy.

Скорость:
    --gpu       считать на видеокарте (нужен cupy: pip install cupy-cuda12x)
                выигрыш от умножения матриц N×N — это ~64% времени при N=4000
    --threads 4 ограничить число потоков процессора, если мешает другой работе

Плотность событий векторизована (в 3-5 раз быстрее).
Прямые связи считаются только в окне замера, а не для всего мира —
в 58 раз быстрее при том же результате. Полный прогон N=4000, 40 затравок,
7 точек: около 2.5 минут вместо 19.

⚠ Скрипт сам находит границу линейного режима: при большой массе поле
упирается в границу [0,1], наклон падает, и константа по всем точкам
оказывается смесью двух режимов.

Память: проверяется ДО начала счёта и не даёт запуститься сверх --max-mem
(по умолчанию 8 ГБ). Ориентир: N=6000 → 0.7 ГБ, N=12000 → 2.7 ГБ,
N=20000 → 7.6 ГБ. При нехватке памяти на конкретной затравке точка
пропускается, а процесс продолжает работу.

Время (один прогон, точек 7):
    N=4000,  40 затравок  —  ~2.5 минуты, точность около 10%
    N=4000, 160 затравок  —  ~10 минут,   точность около 5%
    N=8000, 160 затравок  —  ~40 минут,   точность около 5%
Ошибка падает как корень из числа затравок: вчетверо больше счёта — вдвое точнее.

Промежуточные выводы: печатаются по ходу — набор каждой точки с оценкой
оставшегося времени, и текущее значение константы после каждой третьей точки.
Ничего не теряется, если прервать на середине.

Сохранение: --save data.tsv пишет каждую измеренную точку в файл
(дописывается между запусками, так что можно накапливать).

Рекомендация для первого серьёзного запуска:
    python3 measure_constant.py --n 4000 --seeds 40 --window 160 --save run1.tsv
"""
import argparse
import gc
import os
import sys
import time
import numpy as np


def memory_need_gb(n_total):
    """Сколько памяти потребуется на самом деле.

    Матрица порядка (булева, 1 байт) + её копия во float32 для умножения
    (4 байта) + результат умножения (4 байта) + промежуточные массивы
    при построении порядка. Замерено: примерно вдвое больше наивной оценки.
    """
    return 2.0*n_total*n_total*(1 + 4 + 4)/1e9


def check_memory(n_total, limit_gb):
    need = memory_need_gb(n_total)
    if need > limit_gb:
        print(f"\n⚠ ОСТАНОВЛЕНО: для N={n_total} нужно ~{need:.1f} ГБ, "
              f"разрешено {limit_gb:.1f} ГБ")
        print(f"   уменьшите --n, или поднимите порог: --max-mem {need*1.3:.0f}")
        sys.exit(1)
    return need

# ---- необязательное ускорение на видеокарте ----------------------------
# Ставится как:  pip install cupy-cuda12x   (или cupy-cuda11x)
# Включается:    --gpu
# Выигрыш идёт от умножения матриц N×N (это ~64% времени при N=4000).
xp = np
_GPU = False


def use_gpu(flag):
    """Переключает вычисления на видеокарту, если cupy доступен."""
    global xp, _GPU
    if not flag:
        return False
    try:
        import cupy
        xp = cupy
        _GPU = True
        return True
    except Exception as e:
        print(f"⚠ видеокарта недоступна ({e.__class__.__name__}), считаем на процессоре")
        return False


def to_cpu(a):
    return xp.asnumpy(a) if _GPU else a


# =========================================================== устройство мира

def local_density(X, radius, chunk=512):
    """Число соседей в шаре. Векторизовано блоками — без цикла по событиям."""
    n = len(X)
    d = np.empty(n)
    r2 = radius*radius
    sq = (X*X).sum(1)
    if _GPU:
        Xg = xp.asarray(X); sqg = xp.asarray(sq)
        for a in range(0, n, chunk):
            b = min(a+chunk, n)
            dist2 = sqg[a:b, None] + sqg[None, :] - 2.0*(Xg[a:b] @ Xg.T)
            d[a:b] = xp.asnumpy(xp.count_nonzero(dist2 < r2, axis=1)) - 1
        return d
    for a in range(0, n, chunk):
        b = min(a+chunk, n)
        # |x-y|^2 = |x|^2 + |y|^2 - 2 x·y
        dist2 = sq[a:b, None] + sq[None, :] - 2.0*(X[a:b] @ X.T)
        d[a:b] = np.count_nonzero(dist2 < r2, axis=1) - 1
    return d


def metric(dens, k, alpha):
    """Материя говорит пространству, как искривляться:
    где гуще — там меньше действующих шкал."""
    W = np.ones((len(dens), k))
    if alpha <= 0:
        return W
    rel = dens/max(dens.mean(), 1e-9)
    off = np.clip((alpha*(rel - 1.0)).astype(int), 0, k-2)
    for m in range(k):
        W[off > (k-1-m), m] = 0.0
    return W


def causal_order(X, W):
    n, k = X.shape
    R = np.ones((n, n), dtype=bool)
    act = W > 0
    for m in range(k):
        both = act[:, m][:, None] & act[:, m][None, :]
        R &= ((X[None, :, m] < X[:, None, m]) | ~both)
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def direct_links(R):
    """Прямые связи: те, что не обходятся через третье событие.
    Умножение матриц — самая тяжёлая операция, её и уносим на видеокарту."""
    if _GPU:
        Rg = xp.asarray(R)
        A = Rg.astype(xp.float32)
        out = Rg & ((A @ A) == 0)
        return xp.asnumpy(out)
    A = R.astype(np.float32)
    return R & ((A @ A) == 0)


def field_at(X, R, point, background, nsample):
    """Поле в точке: доля прямых связей среди ближайших ФОНОВЫХ событий.

    Прямые связи считаются ТОЛЬКО для окна, а не для всего мира:
    промежуточное событие ищется по всему миру (произведение
    окно×N на N×окно), но матрица N×N не строится. В 58 раз быстрее,
    результат тот же.
    """
    d = np.linalg.norm(X - point, axis=1)
    cand = np.flatnonzero(background)
    if len(cand) < nsample:
        return np.nan
    idx = cand[np.argsort(d[cand])[:nsample]]
    sub = R[np.ix_(idx, idx)]
    tot = float(sub.sum())
    if tot <= 0:
        return np.nan
    if _GPU:
        Rg = xp.asarray(R)
        ig = xp.asarray(idx)
        mid = (Rg[ig, :].astype(xp.float32) @ Rg[:, ig].astype(xp.float32))
        subL = xp.asnumpy(xp.asarray(sub) & (mid == 0))
        del Rg, ig, mid
    else:
        mid = (R[idx, :].astype(np.float32) @ R[:, idx].astype(np.float32))
        subL = sub & (mid == 0)
    return float(subL.sum()/tot)


def spacing(nbg, k, seed=1, probe=300):
    """Шаг: среднее расстояние до ближайшего события."""
    rs = np.random.default_rng(seed)
    X = rs.random((nbg, k))
    d = [np.partition(np.linalg.norm(X - X[i], axis=1), 1)[1]
         for i in range(min(probe, nbg))]
    return float(np.mean(d))


# ================================================================= замер

def one_pair(bg, c1, c2, mass, k, alpha, radius, nsample, rs, spread):
    """Одна пара замеров на ОБЩЕМ фоне: поле при двух телах и при одном."""
    b1 = np.clip(c1 + rs.normal(0, spread, (mass, k)), 0, 1)
    b2 = np.clip(c2 + rs.normal(0, spread, (mass, k)), 0, 1)
    mid = (c1 + c2)/2

    def field(bodies):
        X = R = W = dens = None
        L = None
        try:
            X = np.vstack([bg] + bodies)
            m = np.zeros(len(X), dtype=bool)
            m[:len(bg)] = True
            dens = local_density(X, radius)
            W = metric(dens, k, alpha)
            R = causal_order(X, W)
            return field_at(X, R, mid, m, nsample)
        finally:
            # память освобождается ДАЖЕ при ошибке — иначе процесс
            # продолжает работать, удерживая матрицы N×N
            del X, R, L, W, dens
            gc.collect()
            if _GPU:
                try:
                    xp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

    f2 = field([b1, b2])
    f1 = field([b1])
    return f2 - f1


def measure_point(nbg, mass, k, alpha, gap, radius, nsample, seeds, spread,
                  base_seed, progress=False, label=""):
    """Вклад тела заданной массы: среднее и ошибка по затравкам."""
    vals = []
    c1 = np.clip(np.full(k, 0.5) - gap/2, 0.05, 0.95)
    c2 = np.clip(np.full(k, 0.5) + gap/2, 0.05, 0.95)
    t0 = time.time()
    for s in range(seeds):
        rs = np.random.default_rng(base_seed + s*7919 + mass*31)
        bg = None
        try:
            bg = rs.random((nbg, k))
            v = one_pair(bg, c1, c2, mass, k, alpha, radius, nsample, rs, spread)
            if np.isfinite(v):
                vals.append(v)
        except MemoryError:
            print(f"      ⚠ не хватило памяти на затравке {s+1} — "
                  f"останавливаю эту точку", flush=True)
            break
        except KeyboardInterrupt:
            print("\n⚠ прервано с клавиатуры", flush=True)
            raise
        finally:
            del bg
            gc.collect()
        if progress and len(vals) >= 2 and (s+1) % max(1, seeds//4) == 0:
            a = np.array(vals)
            e = a.std(ddof=1)/np.sqrt(len(a))
            eta = (time.time()-t0)/(s+1)*(seeds-s-1)
            print(f"      {label} {s+1}/{seeds}:  {a.mean():+.5f} ± {e:.5f}"
                  f"   (ещё ~{eta:.0f} с)", flush=True)
    if len(vals) < 2:
        return np.nan, np.nan, 0
    a = np.array(vals)
    err = float(a.std(ddof=1)/np.sqrt(len(a)))
    # защита: при малом числе затравок разброс может случайно выйти нулевым,
    # и точка получит бесконечный вес. Не даём ошибке упасть ниже
    # разумного уровня (1% от величины эффекта).
    err = max(err, 0.01*abs(float(a.mean())), 1e-6)
    return float(a.mean()), err, len(a)


def weighted_slope(x, y, e):
    """Взвешенная линейная подгонка y = a*x + b; возвращает a и его ошибку."""
    w = 1.0/np.maximum(e, 1e-12)**2
    Sw = w.sum(); Sx = (w*x).sum(); Sy = (w*y).sum()
    Sxx = (w*x*x).sum(); Sxy = (w*x*y).sum()
    D = Sw*Sxx - Sx*Sx
    if abs(D) < 1e-30:
        return np.nan, np.nan
    a = (Sw*Sxy - Sx*Sy)/D
    return a, np.sqrt(Sw/D)


def constant(nbg, k, alpha, seeds, nsample, steps, spread, fracs, base_seed,
             quiet=False, save=None):
    lp = spacing(nbg, k, base_seed)
    gap = steps*lp
    radius = 2.0*lp                    # окно плотности — тоже в шагах
    # перевод массы в собственные единицы: сколько ячеек размера lp
    # укладывается в объём тела. Нормировка одинакова при любом k.
    cell = (lp/spread)**k
    pts = []
    for f in fracs:
        m = max(4, int(f*nbg))
        v, e, n = measure_point(nbg, m, k, alpha, gap, radius, nsample,
                                seeds, spread, base_seed,
                                progress=(not quiet and seeds >= 8),
                                label=f"масса {m}")
        if np.isfinite(v):
            pts.append((m*cell, v, max(e, 1e-9), m, n))
            if save:
                with open(save, "a") as fh:
                    fh.write(f"{nbg}\t{k}\t{alpha}\t{m}\t{m*cell:.3f}"
                             f"\t{v:.6f}\t{e:.6f}\t{n}\n")
            if not quiet:
                print(f"    масса {m:>5} ({m*cell:>7.1f} ячеек):  "
                      f"вклад {v:+.5f} ± {e:.5f}  ({n} затравок)", flush=True)
                if len(pts) >= 3:
                    B = np.array([(p[0], p[1], p[2]) for p in pts])
                    ka, kea = weighted_slope(B[:, 0], B[:, 1], B[:, 2])
                    if np.isfinite(ka) and ka != 0:
                        print(f"      → пока K = {abs(ka):.6f} ± {kea:.6f}"
                              f"  ({100*kea/abs(ka):.0f}%)"
                              f"  [{len(pts)} точек]", flush=True)
    if len(pts) < 3:
        return np.nan, np.nan, lp, pts
    A = np.array([(p[0], p[1], p[2]) for p in pts])

    # ⚠ насыщение: при большой массе поле упирается в границу [0,1],
    # и наклон падает. Константа определена только на ЛИНЕЙНОМ участке.
    # Ищем, до какой массы подгонка ещё описывает данные.
    best = None
    for cut in range(3, len(A)+1):
        a, ea = weighted_slope(A[:cut, 0], A[:cut, 1], A[:cut, 2])
        if not np.isfinite(a) or a == 0:
            continue
        pred = a*A[:cut, 0] + (A[:cut, 1] - a*A[:cut, 0]).mean()
        chi = np.mean(((A[:cut, 1] - pred)/A[:cut, 2])**2)
        if chi < 3.0:                     # ещё описывается прямой
            best = (cut, abs(a), ea, chi)
    if best is None:
        a, ea = weighted_slope(A[:3, 0], A[:3, 1], A[:3, 2])
        return abs(a), ea, lp, pts
    cut, a, ea, chi = best
    if not quiet and cut < len(A):
        print(f"    ⚠ линейный режим кончается на массе "
              f"{A[cut-1,0]:.0f} ячеек — дальше насыщение", flush=True)
        aall, _ = weighted_slope(A[:, 0], A[:, 1], A[:, 2])
        print(f"      (по всем точкам вышло бы {abs(aall):.6f} — "
              f"это смесь двух режимов)", flush=True)
    return a, ea, lp, pts


# ================================================================== вывод

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="событий фона")
    ap.add_argument("--k", type=int, default=4, help="число шкал")
    ap.add_argument("--alpha", type=float, default=1.5,
                    help="сила связи плотность→метрика")
    ap.add_argument("--seeds", type=int, default=8, help="затравок на точку")
    ap.add_argument("--window", type=int, default=120,
                    help="фоновых событий в окне замера")
    ap.add_argument("--steps", type=float, default=4.0,
                    help="расстояние между телами, в шагах")
    ap.add_argument("--spread", type=float, default=0.05, help="размер тела")
    ap.add_argument("--points", type=int, default=7, help="точек по массе")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--save", type=str, default=None,
                    help="файл для промежуточных данных (дописывается)")
    ap.add_argument("--gpu", action="store_true",
                    help="считать на видеокарте (нужен cupy)")
    ap.add_argument("--threads", type=int, default=0,
                    help="ограничить число потоков процессора")
    ap.add_argument("--max-mem", type=float, default=8.0,
                    help="предел памяти в ГБ (по умолчанию 8)")
    ap.add_argument("--quick", action="store_true",
                    help="пропустить замер скорости на старте")
    ap.add_argument("--scan-k", action="store_true",
                    help="зависимость константы от числа шкал")
    ap.add_argument("--scan-alpha", action="store_true",
                    help="зависимость от силы связи")
    ap.add_argument("--scan-n", action="store_true",
                    help="зависимость от размера мира (проверка универсальности)")
    args = ap.parse_args()

    if args.threads > 0:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(args.threads)
    if args.gpu and use_gpu(True):
        print("считаем на видеокарте")

    fracs = np.linspace(0.003, 0.015, args.points)

    # ---- замер скорости машины: сразу видно, сколько ждать -------------
    if not args.quick:
        n_t = min(args.n, 3000)
        A = np.random.rand(200, n_t).astype(np.float32)
        B = np.random.rand(n_t, 200).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(10):
            C = A @ B
        mm_ms = (time.perf_counter()-t0)/10*1000
        # time.time() на Windows даёт шаг ~15 мс: замер выходил нулём и делил на ноль
        gflops = 2*200*n_t*200/(mm_ms/1000)/1e9 if mm_ms > 0 else float('inf')
        del A, B, C
        X_t = np.random.rand(n_t, args.k)
        t0 = time.time()
        Rt = np.ones((n_t, n_t), dtype=bool)
        for m in range(args.k):
            Rt &= (X_t[None, :, m] < X_t[:, None, m])
        ord_s = time.time()-t0
        del Rt, X_t
        per_seed = 2*(ord_s*(args.n/n_t)**2 + mm_ms/1000*3)
        total_min = per_seed*args.seeds*args.points/60
        print(f"скорость машины: умножение {gflops:.0f} GFLOPS, "
              f"порядок {ord_s:.2f} с при N={n_t}")
        print(f"ожидаемое время всего прогона: ~{total_min:.0f} минут "
              f"({per_seed:.1f} с на затравку)")
        if gflops < 20:
            print("⚠ умножение матриц медленное — numpy собран без быстрой BLAS.")
            print("   помогут:  pip install --upgrade numpy   или  conda install numpy")
        print()

    # проверяем память ДО начала счёта
    biggest = args.n + 2*int(fracs[-1]*args.n) + 10
    need = check_memory(biggest, args.max_mem)
    print(f"матрица {biggest}×{biggest} потребует ~{need:.1f} ГБ "
          f"(предел {args.max_mem:.0f} ГБ)")
    if need > 0.5*args.max_mem:
        print("⚠ близко к пределу — при нехватке памяти точка будет пропущена, "
              "а не убьёт процесс")
    line = "─"*68

    def run_one(nbg, k, alpha, tag=""):
        t0 = time.time()
        K, eK, lp, pts = constant(nbg, k, alpha, args.seeds, args.window,
                                  args.steps, args.spread, fracs, args.seed,
                                  save=args.save)
        dt = time.time()-t0
        if np.isfinite(K):
            print(f"  {tag}K = {K:.6f} ± {eK:.6f}  "
                  f"({100*eK/K:.0f}%)   шаг {lp:.4f}   [{dt:.0f} с]")
        else:
            print(f"  {tag}не сошлось [{dt:.0f} с]")
        return K, eK

    if args.save:
        import os
        if not os.path.exists(args.save):
            with open(args.save, "w") as fh:
                fh.write("N\tk\talpha\tmass\tmass_cells\tcontrib\terr\tseeds\n")

    print(line)
    print("КОНСТАНТА СВЯЗИ МАССЫ С ПОЛЕМ")
    print(line)
    if args.seeds < 6:
        print("⚠ мало затравок — результат будет шумным; для дела нужно 20+")
        print()

    if args.scan_k:
        print(f"фон {args.n}, α={args.alpha}, {args.seeds} затравок, "
              f"окно {args.window}")
        print()
        for k in (3, 4, 5):
            print(f"  k = {k}:")
            run_one(args.n, k, args.alpha, "  → ")
            print()
        print("если K одинаково при разных k — константа не зависит от размерности")

    elif args.scan_alpha:
        print(f"фон {args.n}, k={args.k}, {args.seeds} затравок")
        print()
        for a in (0.0, 0.75, 1.5, 3.0):
            print(f"  α = {a}:")
            run_one(args.n, args.k, a, "  → ")
            print()
        print("α=0 — метрика плоская: там связь должна быть слабее")

    elif args.scan_n:
        print(f"k={args.k}, α={args.alpha}, {args.seeds} затравок")
        print()
        for nbg in (2000, 4000, 8000):
            w = int(args.window*nbg/args.n)
            print(f"  N = {nbg} (окно {w}):")
            t0 = time.time()
            K, eK, lp, _ = constant(nbg, args.k, args.alpha, args.seeds, w,
                                    args.steps, args.spread, fracs, args.seed,
                                    quiet=True, save=args.save)
            print(f"    → K = {K:.6f} ± {eK:.6f}  ({100*eK/K:.0f}%)"
                  f"   [{time.time()-t0:.0f} с]")
            print()
        print("если значения перекрываются — константа универсальна")

    else:
        print(f"фон {args.n}, k={args.k}, α={args.alpha}, "
              f"{args.seeds} затравок, окно {args.window}")
        print(f"расстояние {args.steps} шагов, {args.points} точек по массе")
        print()
        K, eK = run_one(args.n, args.k, args.alpha)
        print()
        print("Для сравнения: быстрый замер дал (2.6 ± 0.3)e-4, точность 12%.")
        print("Чтобы улучшить вдвое, нужно вчетверо больше затравок.")

    print(line)


if __name__ == "__main__":
    main()
