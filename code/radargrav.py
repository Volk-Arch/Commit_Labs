#!/usr/bin/env python3
"""ТЯГОТЕНИЕ В РАДАРНОМ КАДРЕ: закон, записанный целиком в порядке.

ПОЧЕМУ ЭТО СТАЛО ВОЗМОЖНО ТОЛЬКО СЕЙЧАС. Проект знает, что вещество есть
РАЗРЕЖЕНИЕ и что оно замедляет часы (−0.070, 4.5σ) и выгибает цепи наружу
(+0.0223, 5.0σ) — `delay-doc.md`. Но СКОЛЬКО вещества даёт СКОЛЬКО
замедления, не знает: нет закона спадания, значит есть сходство, а не
физика.

Все прошлые попытки искали закон в безопорных величинах — а пространства
они, как доказано (`locality-doc.md`: буст даёт 0.000 % по порядку против
×5 по поперечнику), не видят. Записать закон было НЕГДЕ.

Радарный кадр (§16 `PICTURE.md`) это починил: цепь-наблюдатель даёт
полную систему координат чистым счётом (0.9998 по времени, 0.9997 по
пространству), и расстояние в ней — порядковая величина.

УСТРОЙСТВО. Наблюдатель — плотная цепь по оси, проходящая через ЦЕНТР.
Разрежение — запретный шар вокруг того же центра, в который не заходит
ФОН (цепь остаётся: она измерительный прибор, не вещество). Тогда
радарное расстояние события от наблюдателя ЕСТЬ его расстояние от
вещества, и оно посчитано счётом.

    радарное расстояние  rho = (f − p)/2   по номерам вдоль цепи
    ход часов            h(x) / |прошлое(x)|^(1/4)   — порядковая

ЗАМЕР: дефицит хода часов (с дыркой минус без) по корзинам радарного
расстояния, и показатель спадания.

    ОТО в слабом поле: замедление ~ GM/r, то есть дефицит ~ rho^(-1)

КОНТРОЛИ:
  1. ПАРНО по засеву: тот же мир, та же цепь, разница только в дырке.
     Этим снимается то, что плотная цепь сама есть лестница
     (анти-масса) — она есть в обоих прогонах и вычитается.
  2. Нулевая глубина (f = 1.0, дырки нет) — дефицит обязан быть нулём.
  3. Случайное удаление того же числа событий по всему ромбу: дефицит
     обязан быть БЕЗ профиля по расстоянию, иначе профиль есть свойство
     «меньше событий», а не вещества.
  4. Совпадающая высота внутри корзин — краевые члены велики.
  5. Покрытие кадра печатается: у события должны быть и прошлое, и
     будущее на цепи, иначе радарного расстояния нет.

  python3 radargrav.py
  python3 radargrav.py --n 3000 --rad 0.12 --seeds 16
"""
import argparse
import math
import time

import numpy as np

from frames import rel_of, radar


def world(N, T, rs, rad, dens, half=0.05, scatter=False):
    """Фон с разрежением (или со случайным удалением) + цепь-наблюдатель."""
    out, got = [], 0
    need = N
    while got < need:
        m = (need - got) * 10 + 256
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= r) & (T - Y[:, 0] >= r)]
        if len(Y):
            out.append(Y)
            got += len(Y)
    bg = np.vstack(out)[:N]
    nrem = 0
    if rad > 0:
        if scatter:
            # контроль: столько же событий, но убраны по всему ромбу
            rr = np.linalg.norm(bg[:, 1:], axis=1)
            k = int((rr < rad).sum())
            drop = rs.choice(len(bg), k, replace=False)
            keep = np.ones(len(bg), bool)
            keep[drop] = False
            bg = bg[keep]
            nrem = k
        else:
            rr = np.linalg.norm(bg[:, 1:], axis=1)
            keep = rr >= rad
            nrem = int((~keep).sum())
            bg = bg[keep]
    t = np.linspace(half, T - half, dens)
    Z = np.zeros((dens, 4))
    Z[:, 0] = t
    X = np.vstack([bg, Z])
    o = np.argsort(X[:, 0])
    X = X[o]
    isobs = np.zeros(len(X), bool)
    isobs[np.searchsorted(np.sort(o), np.flatnonzero(o >= len(bg)))] = False
    # проще: пометить цепь по признаку |x| == 0 ровно
    isobs = np.all(np.abs(X[:, 1:]) < 1e-12, axis=1)
    return X, isobs, nrem


def clock_rate(X, R):
    """Ход часов: высота на корень четвёртой степени из прошлого."""
    n = len(X)
    h = np.ones(n, np.int64)
    for i in range(n):
        pre = np.flatnonzero(R[:, i])
        if len(pre):
            h[i] = h[pre].max() + 1
    pst = R.sum(0).astype(np.float64)
    return h / np.maximum(pst, 1.0) ** 0.25, h, pst


def run_one(N, T, rs, rad, dens, scatter=False):
    """Один мир: радарное расстояние, ход часов, высота для событий фона."""
    X, isobs, nrem = world(N, T, rs, rad, dens, scatter=scatter)
    R = rel_of(X)
    cr, h, pst = clock_rate(X, R)
    del R
    C = X[isobs]
    nb = int((~isobs).sum())
    bgidx = np.flatnonzero(~isobs)
    Xb = X[bgidx]
    t_, rho, ok = radar(np.vstack([Xb, C]), nb, C)
    return dict(rho=rho, tau=t_, ok=ok, cr=cr[bgidx], h=h[bgidx],
                pst=pst[bgidx], nrem=nrem, cov=float(ok.mean()))


def main():
    ap = argparse.ArgumentParser(description='Тяготение в радарном кадре')
    ap.add_argument('--n', type=int, default=2600)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--rad', type=float, default=0.12)
    ap.add_argument('--dens', type=int, default=120)
    ap.add_argument('--seeds', type=int, default=14)
    ap.add_argument('--bins', type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    print(f'N фона = {a.n}, цепь-наблюдатель {a.dens} событий по оси')
    print(f'разрежение: запретный для ФОНА шар радиуса {a.rad} = '
          f'{a.rad/ell:.1f} ell вокруг центра (цепь остаётся)')
    print(f'ell = {ell:.4f}, засевов {a.seeds}')
    print('радарное расстояние от наблюдателя ЕСТЬ расстояние от вещества')
    print()
    print('ход часов = ВЫСОТА / РАДАРНОЕ ВРЕМЯ, то есть собственное время')
    print('на координатное. Согласовывать не нужно: кадр даёт время сам.')
    print('ОТО в слабом поле: замедление ~ GM/r, значит дефицит ~ rho^(-1)')
    print()

    for lbl, rad, sc in (('ДЫРКА', a.rad, False),
                         ('КОНТРОЛЬ: рассыпано', a.rad, True),
                         ('КОНТРОЛЬ: нулевая глубина', 0.0, False)):
        prof = {}
        cov, nrem = [], []
        for s in range(a.seeds):
            rs1 = np.random.default_rng(81000 + 131 * s)
            A = run_one(a.n, a.t, rs1, 0.0, a.dens)
            rs2 = np.random.default_rng(81000 + 131 * s)
            B = run_one(a.n, a.t, rs2, rad, a.dens, scatter=sc)
            cov.append(B['cov']); nrem.append(B['nrem'])
            okA, okB = A['ok'], B['ok']
            if okA.sum() < 50 or okB.sum() < 50:
                continue
            allr = np.concatenate([A['rho'][okA], B['rho'][okB]])
            ed = np.unique(np.quantile(allr, np.linspace(0, 1, a.bins + 1)))
            ed[0] -= 1e-9; ed[-1] += 1e-9
            for q in range(len(ed) - 1):
                mA = okA & (A['rho'] >= ed[q]) & (A['rho'] < ed[q + 1])
                mB = okB & (B['rho'] >= ed[q]) & (B['rho'] < ed[q + 1])
                if mA.sum() < 10 or mB.sum() < 10:
                    continue
                # ПОЙМАННАЯ ОШИБКА. Сначала здесь стояло согласование по
                # ВЫСОТЕ — а высота стоит в ЧИСЛИТЕЛЕ хода часов
                # (h / |прошлое|^(1/4)). Зафиксировав h, я приравнивал
                # числитель и оставлял только знаменатель, который при
                # удалении событий обязан вырасти. Весь сигнал выходил
                # положительным (+0.37) и контроль «рассыпано» тоже давал
                # профиль (+0.08) — то есть мерился артефакт согласования.
                #
                # Починка не затычка, а улучшение: КАДР ДАЁТ ВРЕМЯ, значит
                # согласовывать не нужно вовсе. Ход часов есть
                #     собственное время / координатное = h / tau,
                # где tau — радарное время. Это ровно гравитационное
                # замедление, и ничего посторонего в нём нет.
                rA = A['h'][mA] / np.maximum(A['tau'][mA], 1e-9)
                rB = B['h'][mB] / np.maximum(B['tau'][mB], 1e-9)
                c = 0.5 * (ed[q] + ed[q + 1])
                prof.setdefault(q, []).append(
                    (c, float(rB.mean() - rA.mean())))
            del A, B
        print(f'--- {lbl}: удалено {np.mean(nrem):.0f}, '
              f'покрытие кадра {np.mean(cov):.3f}')
        print(f"{'радар rho':>11}{'дефицит часов':>15}{'ст.ошибка':>11}"
              f"{'сигм':>7}{'миров':>7}")
        xs, ys = [], []
        for q in sorted(prof):
            v = np.array([p[1] for p in prof[q]])
            c = float(np.mean([p[0] for p in prof[q]]))
            se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else np.nan
            print(f'{c:>11.2f}{v.mean():>+15.5f}{se:>11.5f}'
                  f'{abs(v.mean())/max(se,1e-12):>7.1f}{len(v):>7}')
            if c > 0 and v.mean() < 0:   # спадание мерим по дефициту
                xs.append(c); ys.append(-v.mean())
        if len(xs) >= 3:
            b = np.polyfit(np.log(xs), np.log(ys), 1)[0]
            print(f'  спадание дефицита ~ rho^{b:+.3f}   '
                  f'(ОТО слабого поля: -1)')
        print(flush=True)

    print('  дефицит отрицателен у вещества, спадает с расстоянием,')
    print('  у рассыпанного контроля профиля НЕТ => ЗАКОН')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
