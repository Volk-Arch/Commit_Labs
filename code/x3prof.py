#!/usr/bin/env python3
"""ПРОФИЛИ A_3(v), B_3(v) для `X_3` — закрытие исправления прибора.

Для `X_2` профили выведены и проверены (`design-doc.md`): допущение
`X_2 ~ R_uu` опровергнуто, верное выражение
`X_2 = [A(v²) R_etaeta + B(v²) R_xx] tau²`. У `X_3` моменты
ТРЁХточечные, значит профили СВОИ, и до их вывода вторая наблюдаемая
непригодна.

Алгебра та же по форме. Для конформно-плоских метрик причинность
плоская, вес `w = a^4`, и при наблюдённом числе точек

    1 + X_3 = [ int int int_{x<y<z} w w w ] / [ c_3 (int w)^3 ]

Разложение `w = 1 + phi' delta + (phi''+phi'^2) delta²/2` у середины;
первый порядок уходит по симметрии отражения времени. Остаются моменты
по упорядоченным ТРОЙКАМ:

    S  = <delta²>                              по ромбу
    Q3 = <delta_x² + delta_y² + delta_z²>      по тройкам
    P3 = <delta_x delta_y + delta_x delta_z + delta_y delta_z>

Знаменатель даёт `(1 + f_2 S)^3 ~ 1 + 3 f_2 S`, отсюда `3S`, а не `2S`
как у `X_2`:

    X_3 = 16 vv P3 + (2u + 6vv)(Q3 - 3S)
    A_3 = 8 P3/3 + (2/3)(Q3 - 3S),   B_3 = 8 P3 + 4 (Q3 - 3S)

Сэмплер двухуровневый и точный, без отбрасывания: `x` равномерно в
`I(p,q)`, `y` равномерно в `I(x,q)`, `z` равномерно в `I(y,q)`, вес
тройки — произведение объёмов `tau_xq^d tau_yq^d`.

КОНТРОЛИ:
  1. постоянная `c_3 = <tau_xq^4 tau_yq^4> / T^8` обязана выйти равной
     замкнутой форме `c_k(3)`;
  2. тот же код на ОДНОМ уровне обязан воспроизвести уже выведенные
     `A(0) = -0.02539`, `B(0) = +0.01908` для `X_2`.

  python3 x3prof.py
"""
import math

import numpy as np

from pencil import c_k
from vcal import ball

NB = 5
D = 4


def mk(a, b):
    return a[:, 0] * b[:, 0] - np.einsum('ij,ij->i', a[:, 1:], b[:, 1:])


def rest_diamond(rs, n, T, d=D):
    """Равномерно в ромбе собственного времени T, координаты покоя."""
    z = rs.random(n)
    half = z < 0.5
    y = np.where(half, 2 * z, 2 * (1 - z)) ** (1.0 / d)
    s = np.where(half, 0.5 * y, 1.0 - 0.5 * y) * T
    R = np.minimum(s, T - s)
    return s, ball(rs, n, d - 1, R)


def sub_sample(a, q, rs, d=D):
    """Равномерная точка в ромбе I(a,q); возвращает (точка, tau_aq, маска)."""
    dd = q - a
    t2 = mk(dd, dd)
    ok = t2 > 1e-12
    a, dd, t2 = a[ok], dd[ok], t2[ok]
    tau = np.sqrt(t2)
    m = len(a)
    e0 = dd / tau[:, None]
    B = np.zeros((m, d - 1, d))
    seed = np.zeros((m, d - 1, d))
    for i in range(d - 1):
        seed[:, i, i + 1] = 1.0
    for i in range(d - 1):
        v = seed[:, i, :].copy()
        v -= mk(v, e0)[:, None] * e0
        for j in range(i):
            v -= (-mk(v, B[:, j, :]))[:, None] * B[:, j, :]
        v /= np.sqrt(np.maximum(-mk(v, v), 1e-300))[:, None]
        B[:, i, :] = v
    s, wv = rest_diamond(rs, m, 1.0, d)
    p = a + (s * tau)[:, None] * e0
    for i in range(d - 1):
        p = p + (wv[:, i] * tau)[:, None] * B[:, i, :]
    return p, tau, ok


def moments3(v, n, rs, tau=1.0, d=D):
    """S, Q3, P3 и оценка c_3 для ромба с быстротой v."""
    gam = 1.0 / math.sqrt(1.0 - v * v)

    def de(pt):
        return ((pt[:, 0] - 0.5 * tau) + v * pt[:, 1]) * gam

    s0, w0 = rest_diamond(rs, n, tau, d)
    x = np.column_stack([s0, w0])
    q = np.zeros((n, d)); q[:, 0] = tau
    S = (de(x) ** 2).mean()
    y, txq, ok1 = sub_sample(x, q, rs, d)
    x1, q1 = x[ok1], q[ok1]
    z, tyq, ok2 = sub_sample(y, q1, rs, d)
    x2, y2, txq2 = x1[ok2], y[ok2], txq[ok2]
    wt = (txq2 ** d) * (tyq ** d)
    dx, dy, dz = de(x2), de(y2), de(z)
    Q3 = (wt * (dx ** 2 + dy ** 2 + dz ** 2)).sum() / wt.sum()
    P3 = (wt * (dx * dy + dx * dz + dy * dz)).sum() / wt.sum()
    c3 = wt.mean() / tau ** (2 * d)
    return S, Q3, P3, c3


def moments2(v, n, rs, tau=1.0, d=D):
    """Контроль 2: один уровень — моменты для `X_2`."""
    gam = 1.0 / math.sqrt(1.0 - v * v)

    def de(pt):
        return ((pt[:, 0] - 0.5 * tau) + v * pt[:, 1]) * gam

    s0, w0 = rest_diamond(rs, n, tau, d)
    x = np.column_stack([s0, w0])
    q = np.zeros((n, d)); q[:, 0] = tau
    S = (de(x) ** 2).mean()
    y, txq, ok = sub_sample(x, q, rs, d)
    x1 = x[ok]
    wt = txq ** d
    dx, dy = de(x1), de(y)
    Q = (wt * (dx ** 2 + dy ** 2)).sum() / wt.sum()
    P = (wt * dx * dy).sum() / wt.sum()
    c2 = wt.mean() / tau ** d
    return S, Q, P, c2


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(1618033)
    N = 1500000
    print(f'точек {N} x {NB} блоков на каждое v')
    print()
    print('=' * 78)
    print('КОНТРОЛЬ 1: постоянные c_2 и c_3 тем же сэмплером')
    r2 = np.array([moments2(0.0, N, rs) for _ in range(NB)])
    r3 = np.array([moments3(0.0, N, rs) for _ in range(NB)])
    for nm, got, ex in (('c_2', r2[:, 3], c_k(2)), ('c_3', r3[:, 3], c_k(3))):
        m = got.mean(); e = got.std(ddof=1) / math.sqrt(NB)
        print(f'  {nm}: {m:.7f} +- {e:.7f}, замкнутая форма {ex:.7f}, '
              f'{abs(m-ex)/max(e,1e-30):.1f} сигм')
    print()
    print('КОНТРОЛЬ 2: тот же код на одном уровне — профили `X_2` при v=0')
    S, Q, P = r2[:, 0].mean(), r2[:, 1].mean(), r2[:, 2].mean()
    A2 = 8 * P / 3 + (2.0 / 3.0) * (Q - 2 * S)
    B2 = 8 * P + 4 * (Q - 2 * S)
    print(f'  A = {A2:+.5f} (выведено ранее -0.02539), '
          f'B = {B2:+.5f} (ранее +0.01908)')
    ok = abs(A2 + 0.02539) < 0.0004 and abs(B2 - 0.01908) < 0.0004
    print(f'  -> {"СОВПАЛО" if ok else "НЕ СОВПАЛО, код расходится"}')
    print()
    print('=' * 78)
    print('ПРОФИЛИ ДЛЯ `X_3`')
    print(f"{'v²':>6}{'S':>10}{'Q3':>10}{'P3':>10}"
          f"{'A_3':>14}{'B_3':>12}{'B_3/A_3':>10}{'A_3/A_2':>10}")
    A2TAB = {0.00: -0.02539, 0.15: -0.02877, 0.30: -0.03341, 0.45: -0.04102,
             0.60: -0.05382, 0.75: -0.08255, 0.85: -0.13395}
    out = []
    for v2 in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85):
        v = math.sqrt(v2)
        rr = np.array([moments3(v, N, rs) for _ in range(NB)])
        S, Q3, P3 = rr[:, 0].mean(), rr[:, 1].mean(), rr[:, 2].mean()
        eP = rr[:, 2].std(ddof=1) / math.sqrt(NB)
        A3 = 8 * P3 / 3 + (2.0 / 3.0) * (Q3 - 3 * S)
        B3 = 8 * P3 + 4 * (Q3 - 3 * S)
        out.append((v2, A3, B3))
        print(f'{v2:>6.2f}{S:>10.5f}{Q3:>10.5f}{P3:>10.5f}'
              f'{A3:>+14.5f}{B3:>+12.5f}{B3/A3:>10.3f}'
              f'{A3/A2TAB[v2]:>10.3f}', flush=True)
    print()
    print('ATAB3 = ' + repr([round(a, 5) for _, a, _ in out]))
    print('BTAB3 = ' + repr([round(b, 5) for _, _, b in out]))
    print('''
ЧТЕНИЕ. Столбец `A_3/A_2` — во сколько `X_3` чувствительнее `X_2` к
`R_etaeta`. Если отношение `B_3/A_3` ОТЛИЧАЕТСЯ от `B/A` у `X_2`, то две
наблюдаемые несут РАЗНЫЕ комбинации компонент, и совместная подгонка
разведёт их лучше, чем каждая по себе. Вчера при неверных столбцах
`X_3` оказался бесполезным повтором `X_2`; теперь это проверяется
заново, уже на верных.''')


if __name__ == '__main__':
    main()
