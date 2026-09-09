#!/usr/bin/env python3
"""
РАЗБОР СОХРАНЁННЫХ ДАННЫХ
=========================

Читает файл, который пишет measure_constant.py (--save), и считает
константу — по всем точкам и отдельно по линейному участку.

Работает с оборванным файлом: если прогон прерван, всё посчитанное
до этого момента используется.

Запуск:
    python3 analyze_run.py run2.tsv
    python3 analyze_run.py run2.tsv scan_k.tsv     # несколько файлов
    python3 analyze_run.py *.tsv --group k         # разбить по числу шкал

Зависимости: numpy.
"""
import argparse
import sys
import numpy as np


def read(paths):
    rows = []
    for p in paths:
        try:
            with open(p, encoding="utf-8-sig") as fh:
                head = None
                for line in fh:
                    line = line.strip().replace("\r", "")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if head is None:
                        head = parts
                        continue
                    if len(parts) != len(head):
                        continue
                    try:
                        rows.append(dict(
                            N=int(parts[0]), k=int(parts[1]),
                            alpha=float(parts[2]), mass=int(parts[3]),
                            cells=float(parts[4]), contrib=float(parts[5]),
                            err=float(parts[6]), seeds=int(parts[7])))
                    except ValueError:
                        continue
        except FileNotFoundError:
            print(f"⚠ файл не найден: {p}")
    return rows


def weighted_slope(x, y, e):
    w = 1.0/np.maximum(e, 1e-12)**2
    Sw = w.sum(); Sx = (w*x).sum(); Sy = (w*y).sum()
    Sxx = (w*x*x).sum(); Sxy = (w*x*y).sum()
    D = Sw*Sxx - Sx*Sx
    if abs(D) < 1e-30:
        return np.nan, np.nan, np.nan
    a = (Sw*Sxy - Sx*Sy)/D
    b = (Sxx*Sy - Sx*Sxy)/D
    ea = np.sqrt(Sw/D)
    pred = a*x + b
    chi = np.mean(((y - pred)/e)**2) if len(x) > 2 else np.nan
    return a, ea, chi


def analyse(rows, label=""):
    if len(rows) < 2:
        print(f"  {label}точек {len(rows)} — мало для наклона")
        return
    rows = sorted(rows, key=lambda r: r["cells"])
    x = np.array([r["cells"] for r in rows])
    y = np.array([r["contrib"] for r in rows])
    e = np.array([max(r["err"], 1e-9) for r in rows])

    print(f"  {label}точек {len(rows)}, затравок на точку "
          f"{rows[0]['seeds']}–{rows[-1]['seeds']}")
    print()
    print(f"    {'масса (яч.)':>13}{'вклад':>11}{'ошибка':>10}"
          f"{'вклад/масса':>14}{'z':>7}")
    for r in rows:
        z = abs(r["contrib"])/max(r["err"], 1e-12)
        print(f"    {r['cells']:>13.1f}{r['contrib']:>11.5f}{r['err']:>10.5f}"
              f"{abs(r['contrib'])/r['cells']:>14.6f}{z:>7.1f}")
    print()

    if len(rows) < 3:
        a, ea, _ = weighted_slope(x, y, e)
        if np.isfinite(a):
            print(f"    K (по {len(rows)} точкам) = {abs(a):.6f} ± {ea:.6f}")
            print("    ⚠ двух точек мало: наклон не отличить от шума")
        return

    a_all, e_all, chi_all = weighted_slope(x, y, e)

    # линейный участок: наращиваем, пока прямая описывает данные И
    # пока отношение вклад/масса не начало систематически падать
    # (падение = насыщение: поле упирается в границу [0,1])
    ratio = np.abs(y)/np.maximum(x, 1e-9)
    peak = int(np.argmax(ratio))
    best = None
    for cut in range(3, len(x)+1):
        a, ea, chi = weighted_slope(x[:cut], y[:cut], e[:cut])
        if not (np.isfinite(a) and a != 0):
            continue
        # не заходим дальше точки, где отношение достигло максимума:
        # после неё эффект насыщается и наклон занижается
        if cut - 1 > peak + 1:
            continue
        if chi < 2.0:
            best = (cut, abs(a), ea, chi)

    print(f"    отношение вклад/масса максимально на точке "
          f"{x[peak]:.0f} ячеек — там кончается линейный режим")
    print()
    print(f"    K по ВСЕМ точкам:        {abs(a_all):.6f} ± {e_all:.6f}"
          f"   (χ²/dof = {chi_all:.1f})")
    if best:
        cut, a, ea, chi = best
        print(f"    K по ЛИНЕЙНОМУ участку: {a:.6f} ± {ea:.6f}"
              f"   (χ²/dof = {chi:.1f}, {cut} точек)")
        if cut < len(x):
            print(f"    ⚠ линейный режим кончается на {x[cut-1]:.0f} ячейках "
                  f"— дальше насыщение")
            print(f"      значение по всем точкам ниже в "
                  f"{a/max(abs(a_all),1e-12):.1f} раза — это смесь режимов")
    else:
        print("    ⚠ прямая не описывает данные ни на каком участке")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="файлы .tsv от measure_constant.py")
    ap.add_argument("--group", choices=["k", "alpha", "N", "none"], default="none",
                    help="разбить по параметру")
    args = ap.parse_args()

    rows = read(args.files)
    if not rows:
        print("данных нет")
        sys.exit(1)

    line = "─"*66
    print(line)
    print(f"РАЗБОР: {len(rows)} измеренных точек")
    print(line)

    if args.group == "none":
        # если в файле разные условия — предупредить
        for key in ("N", "k", "alpha"):
            vals = sorted({r[key] for r in rows})
            if len(vals) > 1:
                print(f"⚠ в данных разные {key}: {vals} — "
                      f"добавьте --group {key}")
        print()
        analyse(rows)
    else:
        key = args.group
        for v in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == v]
            print(f"{key} = {v}:")
            analyse(sub, "  ")

    print(line)


if __name__ == "__main__":
    main()
