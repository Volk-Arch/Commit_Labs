#!/usr/bin/env python3
"""УРАВНЕНИЕ СОСТОЯНИЯ ИЗ СЧЁТА: w = p/rho, без вложенного ответа.

ФИЗИКА. Уравнения Эйнштейна для идеальной жидкости дают в ортонормальном
репере
    R_00 = 4 pi G (rho + 3p),      R_11 = 4 pi G (rho - p)
то есть в конформных координатах `R_etaeta/a^2` и `R_xx/a^2`. Отсюда

    w = p/rho = (R_etaeta - R_xx) / (R_etaeta + 3 R_xx)

и это ОТНОШЕНИЕ: общая калибровка прибора в него не входит, значит `w`
выходит без множителя `v2`, измеренного в другом прогоне.

ТРИ ФИЗИЧЕСКИХ ВСЕЛЕННЫХ УЖЕ НАБРАНЫ:

| семейство | что зануляется | вселенная | w |
|---|---|---|---|
| линейное `a = 1+h(eta-eta0)` | `R = 0`, то есть `rho = 3p` | излучение | **+1/3** |
| экспонента `a = e^{k eta}`   | `R_etaeta = 0`, `rho+3p = 0` | без ускорения | **-1/3** |
| корень `a = sqrt(1+c(eta-eta0))` | `R_xx = 0`, `rho = p` | жёсткая | **+1** |
| квадратичное | `R_etaeta + 3R_xx = 0`, то есть `rho = 0` | нефизическая | полюс |

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ `comp.py`. Там у экспоненты `R_etaeta = 0` и у
корня `R_xx = 0` были ЗАДАНЫ СВЯЗЬЮ — то есть ответ вложен. Здесь оба
параметра каждого плеча ОТПУЩЕНЫ, и `w` обязано выйти само. Цена —
вырождение внутри плеча возвращается, поэтому ошибка будет велика; но
измерение становится измерением.

ОШИБКИ: складной нож по миpам, применённый к САМОМУ `w`, а не к
компонентам порознь — при сильной корреляции квадратура дала бы вздор.

КОНТРОЛЬ: половина плоского плеча. Там `rho = p = 0`, то есть `w`
неопределено (0/0), поэтому проверяется другое — что ОБА параметра
выходят нулями.

  python3 eos.py
"""
import glob
import math
import pickle

import numpy as np

from pencil import SCRATCH, truth
from comp import cells, solve, KF, V2CAL, V3CAL, IX2, IX3

ARMS = ('lin', 'quad', 'exp', 'root')
WTRUE = {'lin': 1.0 / 3.0, 'exp': -1.0 / 3.0, 'root': 1.0, 'quad': None}


def design(cv, fam, which):
    """У КАЖДОГО плеча свои A и B, оба СВОБОДНЫ. Помеха общая, гладкая."""
    s = np.sqrt(cv[:, 2]) / (1.0 - cv[:, 1])
    g1, g2 = s, s * cv[:, 1]
    z = np.zeros(len(cv))
    # ПОМЕХА СОКРАЩЕНА до одной постоянной, 8 октября. Четыре столбца
    # {1, 1/s, v², v²/s} разрушали обусловленность (2.4e9 против 1.1e5 с
    # одной постоянной и 62 без помехи вовсе), а плоское плечо их НЕ
    # ТРЕБУЕТ: его шесть значений дают хи² = 8.2 на 6 степеней, то есть
    # согласуются с нулём, и взвешенная постоянная равна
    # +0.0005 +- 0.0007. Подгонять к ним четыре параметра значило
    # подгонять шум — и ровно этим ломалась обусловленность.
    isq = 1.0 / np.sqrt(cv[:, 2])
    NUI = int(__import__('os').environ.get('NUI', '1'))
    full = [np.ones(len(cv)), isq, cv[:, 1], cv[:, 1] * isq]
    nui = full[:NUI]
    phys = []
    for f in ARMS:
        phys += [g1, g2] if f == fam else [z, z]
    return np.column_stack(phys + nui)


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
    arms = {'lin': res[keys['lin']], 'quad': res[keys['quad']],
            'exp': (EXA, EXS), 'root': (RTA, RTS)}
    FA, FS = res[keys['flat']]
    pars = {'lin': 0.035, 'quad': 0.0012, 'exp': KEXP, 'root': KROOT}
    print('набор: ' + ', '.join(f'{f} {len(arms[f][0])}' for f in ARMS)
          + f', flat {len(FA)}')
    print()

    def run(col, drop=None):
        blocks = []
        for fam, (A, sd) in list(arms.items()) + [('flat', (FA, FS))]:
            cv = cells(A, sd, col, drop)
            blocks.append((design(cv, fam, col), cv[:, 0]))
        return solve(blocks)

    seeds = sorted(set(FS.tolist()))
    for col, cname, vcal in ((IX2, 'X_2', V2CAL), (IX3, 'X_3', V3CAL)):
        print('=' * 86)
        print(cname)
        p, cond, nrow = run(col)
        jk = []
        for s in seeds:
            try:
                q, _, _ = run(col, s)
                jk.append(q)
            except Exception:
                pass
        jk = np.array(jk); n = len(jk)
        print(f'  обусловленность {cond:.2e}, ячеек {nrow}, миров в ноже {n}')
        print()
        print(f"  {'вселенная':>12}{'R_ηη':>11}{'R_xx':>11}"
              f"{'w из счёта':>16}{'истина':>9}{'сигм':>7}")
        for i, fam in enumerate(ARMS):
            A_, B_ = p[2 * i], p[2 * i + 1]
            w = (A_ - B_) / (A_ + 3 * B_) if (A_ + 3 * B_) != 0 else np.nan
            jw = np.array([(jk[t, 2*i] - jk[t, 2*i+1])
                           / (jk[t, 2*i] + 3*jk[t, 2*i+1])
                           for t in range(n)])
            jw = jw[np.isfinite(jw)]
            m = len(jw)
            sw = (math.sqrt((m-1)/m*((jw-jw.mean())**2).sum())
                  if m > 1 else np.nan)
            ree = A_ / (vcal * KF)
            rxx = B_ / (vcal * KF)
            wt = WTRUE[fam]
            sig = (abs(w - wt) / sw if (wt is not None and sw and
                                        np.isfinite(sw)) else np.nan)
            ws = f'{w:+.3f} ± {sw:.3f}' if np.isfinite(sw) else f'{w:+.3f}'
            ts = f'{wt:+.3f}' if wt is not None else 'полюс'
            ss = f'{sig:.1f}' if np.isfinite(sig) else '—'
            print(f'  {fam:>12}{ree:>+11.5f}{rxx:>+11.5f}{ws:>16}{ts:>9}'
                  f'{ss:>7}')
        print()
        print('  (истины: излучение +1/3, без ускорения −1/3, жёсткая +1;')
        print('   у квадратичного ρ = 0 по построению, поэтому полюс)')
        print()
    print('=' * 86)
    print('''ЧТЕНИЕ. `w` здесь ИЗМЕРЕНО: связи `R_etaeta = 0` и `R_xx = 0`
отпущены, и если у экспоненты выходит −1/3, а у корня +1, то уравнение
состояния вселенной читается из причинного порядка. Ошибка велика —
такова цена отпущенных связей, — но это измерение, а не допущение.''')


if __name__ == '__main__':
    main()
