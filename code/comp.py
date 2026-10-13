#!/usr/bin/env python3
"""КОМПОНЕНТЫ ПОРОЗНЬ: третье семейство ломает вырождение.

ЧТО БЫЛО. Столбцы плана давали `g1 − g2 = sqrt(C1)` ТОЖДЕСТВЕННО (число
обусловленности 3.7e8), и отношение `R_xx/R_etaeta` у линейного
семейства выходило −0.299 ± 0.247 против истины +1/3 — с НЕВЕРНЫМ
знаком. Статистикой это не лечилось.

ЧТО ДОБАВЛЕНО. Третье семейство `a = e^{k eta}`, у которого
`R_etaeta = 0` ТОЖДЕСТВЕННО (проверено символьно от метрики), а
`R_xx = 2k^2`. Его отклик `R_uu = R_xx v^2/(a^2(1-v^2))` обращается в
НУЛЬ при `v = 0`, тогда как у линейного там максимум. Ход по наклону
качественно другой, поэтому вырождение ломается ФОРМОЙ.

МОДЕЛЬ. Для строки с `(C1, v^2)` в семействе `f`:

    X = sqrt(C1)/(1-v^2) * [A_f + B_f v^2] + помеха(C1, v^2)
    A_f = v2cal*KF*R_etaeta(f),   B_f = v2cal*KF*R_xx(f),  KF = sqrt(24/pi)

При ПАРНОМ вычитании плоского плеча помеха сокращается целиком — она
одна и та же функция `(C1, v^2)` в обоих плечах. Значит параметров
остаётся ПЯТЬ: `A_lin, B_lin, A_quad, B_quad, B_exp` (у экспоненты
`A = 0` по построению), и ни одного нуисанса.

ОШИБКИ: складной нож по ЗАСЕВАМ, единица независимости — сприклинг.
КОНТРОЛЬ: половина плоского объявляется семейством; все пять
параметров обязаны выйти нулями.

  python3 comp.py
"""
import glob
import math
import pickle

import numpy as np

from pencil import SCRATCH, VB, truth

KF = math.sqrt(24.0 / math.pi)
V2CAL, V3CAL = -0.0201, -0.0452
C1B = (200, 380, 560, 900)
IC, IX2, IX3, IV = 0, 1, 2, 3


def cells(A, sd, col, drop=None):
    """Средние по ячейкам (полоса v² x полоса C_1), с выбросом засева."""
    m0 = np.ones(len(A), bool) if drop is None else (sd != drop)
    out = []
    for i in range(len(VB) - 1):
        for j in range(len(C1B) - 1):
            m = (m0 & (A[:, IV] >= VB[i]) & (A[:, IV] < VB[i + 1])
                 & (A[:, IC] >= C1B[j]) & (A[:, IC] < C1B[j + 1]))
            if m.sum() < 15:
                out.append((np.nan,) * 3)
            else:
                out.append((A[m, col].mean(), A[m, IV].mean(),
                            A[m, IC].mean()))
    return np.array(out, float)


def design(cv, fam):
    """Столбцы: 5 физических + 4 ГЛАДКОЙ помехи, общей для всех плеч.

    ПОЧЕМУ НЕ ПАРНОЕ ВЫЧИТАНИЕ ПО ЯЧЕЙКАМ. Это ошибка №3 из списка:
    средние ячеек плоского плеча сами шумны (±0.002…0.004) и сравнимы с
    сигналом в полосах МАЛОГО наклона — а именно они определяют
    `A` (то есть `R_etaeta`). Проверено на опыте: поячеечное вычитание
    дало `R_etaeta lin` на 3.0 сигмы выше истины, тогда как гладкая
    помеха давала 1.1. Помеха физически гладкая, значит описывается
    четырьмя числами на всё, а не 54 независимыми средними.
    """
    s = np.sqrt(cv[:, 2]) / (1.0 - cv[:, 1])
    z = np.zeros(len(cv))
    g1, g2 = s, s * cv[:, 1]
    isq = 1.0 / np.sqrt(cv[:, 2])
    nui = [np.ones(len(cv)), isq, cv[:, 1], cv[:, 1] * isq]
    if fam == 'lin':
        phys = [g1, g2, z, z, z, z]
    elif fam == 'quad':
        phys = [z, z, g1, g2, z, z]
    elif fam == 'exp':
        phys = [z, z, z, z, g2, z]        # A = 0 по построению
    elif fam == 'root':
        phys = [z, z, z, z, z, g1]        # B = 0 по построению
    else:                      # flat: A = B = 0 по построению
        phys = [z, z, z, z, z, z]
    return np.column_stack(phys + nui)


def solve(blocks):
    M = np.vstack([b[0] for b in blocks])
    y = np.concatenate([b[1] for b in blocks])
    ok = np.isfinite(y) & np.isfinite(M).all(axis=1)
    M, y = M[ok], y[ok]
    p, *_ = np.linalg.lstsq(M, y, rcond=None)
    return p, np.linalg.cond(M.T @ M), M.shape[0]


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/pencil.pkl', 'rb') as fh:
        res = pickle.load(fh)
    keys = {k[0]: k for k in res}
    def load(tag):
        A, S, k = [], [], None
        for f in sorted(glob.glob(f'{SCRATCH}/{tag}fam_*.pkl')):
            d = pickle.load(open(f, 'rb'))
            A.append(d['A']); S.append(d['sd']); k = d['k']
        return np.vstack(A), np.concatenate(S), k
    EXA, EXS, KEXP = load('exp')
    RTA, RTS, KROOT = load('root')
    print(f"набор: " + ', '.join(
        f'{k} {len(res[keys[k]][0])}' for k in ('flat', 'lin', 'quad'))
        + f', exp {len(EXA)} (k={KEXP}), root {len(RTA)} (c={KROOT})')
    print()

    FA, FS = res[keys['flat']]
    arms = {'lin': res[keys['lin']], 'quad': res[keys['quad']],
            'exp': (EXA, EXS), 'root': (RTA, RTS)}

    def run(col, drop_map=None):
        """Все плечи в ОДНОЙ подгонке; плоское закрепляет гладкую помеху."""
        blocks = []
        allarms = dict(arms); allarms['flat'] = (FA, FS)
        for fam, (A, sd) in allarms.items():
            cv = cells(A, sd, col, None if drop_map is None else drop_map(fam))
            blocks.append((design(cv, fam), cv[:, 0]))
        return solve(blocks)

    for col, cname, vcal in ((IX2, 'X_2', V2CAL), (IX3, 'X_3', V3CAL)):
        print('=' * 84)
        print(cname)
        p, cond, nrow = run(col)
        seeds = sorted(set(FS.tolist()))
        jk = []
        for s in seeds:
            try:
                q, _, _ = run(col, lambda f, s=s: s)
                jk.append(q)
            except Exception:
                pass
        jk = np.array(jk); n = len(jk)
        se = np.sqrt((n - 1) / n * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
        print(f'  обусловленность плана {cond:.1f} (было 3.7e8), '
              f'ячеек {nrow}, засевов в ноже {n}, параметров 9')
        print()
        names = ('R_ηη lin', 'R_xx lin', 'R_ηη quad', 'R_xx quad',
                 'R_xx exp  ⟵', 'R_ηη root ⟵')
        tru = (truth('lin', 0.035)[0], truth('lin', 0.035)[1],
               truth('quad', 0.0012)[0], truth('quad', 0.0012)[1],
               truth('exp', KEXP)[1], truth('root', KROOT)[0])
        print(f"  {'величина':>12}{'из счёта':>13}{'±':>10}{'истина':>11}"
              f"{'сигм':>7}")
        vals, errs = [], []
        for i in range(6):
            v, e = p[i] / (vcal * KF), se[i] / abs(vcal * KF)
            vals.append(v); errs.append(e)
            print(f'  {names[i]:>12}{v:>+13.5f}{e:>10.5f}{tru[i]:>+11.5f}'
                  f'{abs(v-tru[i])/max(e,1e-12):>7.1f}')
        print()
        # отношение из ДВУХ РАЗНЫХ плеч, каждое со своей компонентой:
        rr = vals[4] / vals[5]
        jr = jk[:, 4] / jk[:, 5]
        sr = math.sqrt((n-1)/n*((jr-jr.mean())**2).sum())
        tr = truth('exp', KEXP)[1] / truth('root', KROOT)[0]
        print(f'  НЕЗАВИСИМО: R_xx(exp)/R_ηη(root) = {rr:+.3f} ± {sr:.3f}'
              f'   (истина {tr:+.3f}, {abs(rr-tr)/max(sr,1e-12):.1f} сигм)')
        print()
        for lab, a, b, t in (('lin', 0, 1, 1/3), ('quad', 2, 3, -1/3)):
            jr = jk[:, b] / jk[:, a]
            r = vals[b] / vals[a]
            sr = math.sqrt((n-1)/n*((jr-jr.mean())**2).sum())
            print(f'  R_xx/R_ηη {lab:>4} = {r:+.3f} ± {sr:.3f}'
                  f'   (истина {t:+.3f}, {abs(r-t)/max(sr,1e-12):.1f} сигм)')
        print()
    print('=' * 84)
    print('''КОНТРОЛЬ сдвинут в отдельный прогон: половину плоского
объявить семейством и проверить, что все пять параметров нули.''')


if __name__ == '__main__':
    main()
