#!/usr/bin/env python3
"""ПОЛЕ БД: калибровка на плоском, затем отклик на вещество.

ЧТО ДАЛ ВЕЧЕР (`cancel-doc.md`). Впервые за проект есть ЛОКАЛЬНОЕ поле на
причинном множестве — плотность действия Бенинказы–Доукера

    b_x = 1 − n₀(x) + 9n₁(x) − 16n₂(x) + 8n₃(x)
    n_m(x) = #{y ≺ x : |I(y,x)| = m}

с проверенным КОМПАКТНЫМ носителем: удаление события дальше чем ~2.5
длины дискретности меняет `b_x` ровно на нуль (0.0e+00 на трёх N), а
размер носителя в длинах дискретности стоит при измельчении
(2.47 / 2.56 / 2.58 при N = 800 / 1600 / 3200).

Это единственная локальность, которую лоренц-инвариантность разрешает:
точная по собственному времени. Пространственного расстояния в порядке
нет вовсе (`locality-doc.md`: буст меняет поперечник в 4.88 раза при
расхождении порядковых величин 0.000 %).

И второй вывод вечера, про метод: сокращение живёт в ЗНАКОВОМ СРЕДНЕМ
(|Δb| = 18.4 при ⟨Δb⟩ = 0.5, то есть в 36 раз), значит всякая проверка
на действии обязана быть АНСАМБЛЕВОЙ. Отдельный мир ничего не покажет —
и это задним числом объясняет, почему вечерние замеры по одному миру
молчали.

ЭТАП 1 — КАЛИБРОВКА. В плоском пространстве `R = 0`, значит в глубине
`⟨b_x⟩` обязано быть нулём (всё остальное — краевые члены). Меряем его и
его шум: это и есть чувствительность замера, известная ДО того, как
посажено вещество. Если ⟨b⟩ в глубине не нуль — мерить нечем, и дальше
идти нельзя.

ЭТАП 2 — ОТКЛИК НА ВЕЩЕСТВО. Вещество есть разрежение (`delay-doc.md`:
замедляет часы 4.5σ, выгибает цепи наружу 5.0σ, монотонно по глубине).
Внутри радиуса остаётся доля `f`. Меряем профиль `⟨b_x⟩` по расстоянию до
дырки.

ВСТРОЕННАЯ ПРОВЕРКА, которой не было ни у одного прежнего замера:
**отклик обязан иметь тот же носитель, что и величина.** Если он тянется
дальше 2.5 `ℓ` от края дырки — отвечает не поле, а что-то другое.

КОНТРОЛИ:
  f = 1.0 — дырки нет, разница обязана быть нулём;
  случайное удаление того же числа событий — парно по засеву;
  совпадающая высота — именно её отсутствие погубило замер по дефектам;
  профиль по расстоянию — проверка носителя (см. выше).

  python3 bdfield.py
  python3 bdfield.py --n 3000 --fracs 1.0 0.5 0.0 --seeds 24
"""
import argparse
import math
import time

import numpy as np

from particles import diamond, rel_of


def bd_density(R):
    """b_x = 1 − n0 + 9n1 − 16n2 + 8n3 для каждого события."""
    A = R.astype(np.float32)
    INT = A @ A                      # INT[y,x] = |I(y,x)| при R[y,x]
    W = (-1.0, 9.0, -16.0, 8.0)
    b = np.ones(len(R))
    for m in range(4):
        b += W[m] * ((INT == m) & R).sum(0)
    del A, INT
    return b


def heights(R):
    N = len(R)
    h = np.ones(N, np.int64)
    for i in range(N):               # уже топологический порядок
        pre = np.flatnonzero(R[:, i])
        if len(pre):
            h[i] = h[pre].max() + 1
    return h


def hole_world(N, rad, f, rs):
    """Ромб с разрежением внутри радиуса rad: остаётся доля f."""
    X = diamond(N, rs)
    rr = np.linalg.norm(X[:, 1:], axis=1)
    cen = np.abs(X[:, 0] - 0.5) < 0.5           # весь ромб по времени
    ins = np.flatnonzero((rr < rad) & cen)
    drop = ins[rs.random(len(ins)) > f] if len(ins) else ins
    keep = np.ones(len(X), bool)
    keep[drop] = False
    return X[keep], len(drop)


def main():
    ap = argparse.ArgumentParser(description='Поле БД')
    ap.add_argument('--n', type=int, default=2600)
    ap.add_argument('--rad', type=float, default=0.14)
    ap.add_argument('--fracs', type=float, nargs='+',
                    default=[1.0, 0.6, 0.3, 0.0])
    ap.add_argument('--seeds', type=int, default=16)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    print(f'N = {a.n}, ромб T = 1; длина дискретности ell = {ell:.4f}')
    print(f'носитель поля БД ~ 2.5 ell = {2.5*ell:.4f}')
    print(f'дырка радиуса {a.rad} = {a.rad/ell:.1f} ell')
    print('ВСЁ усредняется по ансамблю — вывод вечера про знаковое среднее')
    print()

    # --- ЭТАП 1: калибровка на плоском
    print('ЭТАП 1: КАЛИБРОВКА. В плоском R = 0, значит ⟨b⟩ в глубине = 0')
    print(f"{'высота':>8}{'событий':>10}{'⟨b⟩':>12}{'ст.ошибка':>12}"
          f"{'сигм от 0':>11}{'ст.откл b':>12}")
    allb, allh = [], []
    for s in range(a.seeds):
        X = diamond(a.n, np.random.default_rng(20100 + 71 * s))
        R = rel_of(X)
        allb.append(bd_density(R))
        allh.append(heights(R))
        del R, X
    B = np.concatenate(allb)
    H = np.concatenate(allh)
    hmax = int(H.max())
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 9), (10, hmax)):
        m = (H >= lo) & (H <= hi)
        if m.sum() < 50:
            continue
        v = B[m]
        se = v.std(ddof=1) / math.sqrt(len(v))
        print(f'{f"{lo}-{hi}":>8}{int(m.sum()):>10}{v.mean():>12.3f}'
              f'{se:>12.3f}{abs(v.mean())/max(se,1e-12):>11.1f}'
              f'{v.std():>12.1f}')
    mid = (H >= 3) & (H <= hmax - 2)
    v = B[mid]
    se = v.std(ddof=1) / math.sqrt(len(v))
    print(f'  ГЛУБИНА (высота 3..{hmax-2}): ⟨b⟩ = {v.mean():+.3f} ± '
          f'{se:.3f}  ({abs(v.mean())/max(se,1e-12):.1f} сигм от нуля), '
          f'разброс отдельного {v.std():.1f}')
    print(f'  чувствительность: на одно событие {v.std():.1f}, '
          f'на {int(mid.sum())} событий {se:.3f}')
    print()

    # --- ЭТАП 2: отклик на вещество
    print('ЭТАП 2: ОТКЛИК НА РАЗРЕЖЕНИЕ, профиль по расстоянию до дырки')
    print('встроенная проверка: отклик обязан умереть за ~2.5 ell от края')
    print()
    edges = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0]) * ell + a.rad
    lbl = [f'{edges[i]-a.rad:.2f}..{edges[i+1]-a.rad:.2f}'
           for i in range(len(edges) - 1)]
    print(f"{'f':>6}{'удал':>6}" +
          ''.join(f'{f"{q} ell":>14}' for q in
                  ('0-1', '1-2', '2-3', '3-5', '5-8')))

    for f in a.fracs:
        acc = {k: [] for k in range(len(edges) - 1)}
        nrem = 0
        for s in range(a.seeds):
            rs = np.random.default_rng(20300 + 71 * s)
            Xh, k = hole_world(a.n, a.rad, f, rs)
            nrem = k
            rs2 = np.random.default_rng(20300 + 71 * s)
            Xc = diamond(a.n, rs2)
            if k > 0:
                ii = np.sort(rs2.choice(len(Xc), len(Xc) - k, replace=False))
                Xc = Xc[ii]
            Rh, Rc = rel_of(Xh), rel_of(Xc)
            bh, bc = bd_density(Rh), bd_density(Rc)
            hh, hc = heights(Rh), heights(Rc)
            rh = np.linalg.norm(Xh[:, 1:], axis=1)
            rc = np.linalg.norm(Xc[:, 1:], axis=1)
            for q in range(len(edges) - 1):
                mh = (rh >= edges[q]) & (rh < edges[q + 1])
                mc = (rc >= edges[q]) & (rc < edges[q + 1])
                if mh.sum() < 8 or mc.sum() < 8:
                    continue
                # совпадающая высота
                num, den = 0.0, 0.0
                for hv in np.unique(hh[mh]):
                    x1 = bh[mh & (hh == hv)]
                    y1 = bc[mc & (hc == hv)]
                    if len(x1) < 2 or len(y1) < 2:
                        continue
                    num += (x1.mean() - y1.mean()) * len(x1)
                    den += len(x1)
                if den > 0:
                    acc[q].append(num / den)
            del Rh, Rc, Xh, Xc
        line = f'{f:>6.2f}{nrem:>6}'
        for q in range(len(edges) - 1):
            v = np.array(acc[q])
            if len(v) < 4:
                line += f'{"—":>14}'
                continue
            se = v.std(ddof=1) / math.sqrt(len(v))
            sg = abs(v.mean()) / se if se > 1e-12 else float('nan')
            line += f'{f"{v.mean():+.1f}/{sg:.1f}σ":>14}'
        print(line, flush=True)

    print()
    print('  отклик силён у края и умирает за 2.5 ell => ОТВЕЧАЕТ ПОЛЕ;')
    print('  тянется дальше => отвечает что-то другое')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
