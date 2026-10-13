#!/usr/bin/env python3
"""ПРОФИЛЬ ДЕФИЦИТА: оба числа на одной кривой. Пункт 0а инвентаря.

В архиве лежат ДВА взаимоисключающих итога одного замера, причём в одном
файле: `ARTICLE.md:335` даёт относительный показатель
**−0.502 ± 0.042** (ньютоновский закон отвергнут на 11.9 сигм), а
`ARTICLE.md:434` — **−1.062 ± 0.060** (согласие с Ньютоном в 1.0 сигмы,
14 миров). Первое мерено в окне `r in [2.5 l, 0.40 R]`, второе по правилу
`r > 2 r_дырки`, то есть `r > 4.4 l`. Окна СОСЕДНИЕ и не перекрываются.

А `exponent-doc.md` установил, что степенного закона нет вовсе: местный
показатель монотонно крутеет от −0.81 до −1.77.

ЗДЕСЬ это проверяется ЗАМЕРОМ, а не сверкой текстов. `theory-doc.md`
выводит из геометрии ромба

    Delta_tau/tau = -a/(t + r)

У такой функции местный показатель равен ТОЧНО

    d ln|Delta| / d ln r = -r/(t + r)

то есть он ОБЯЗАН крутеть от нуля к −1 и проходить через −1/2 при
`r = t`. Если измеренный профиль на это садится с одним параметром `t`,
то оба архивных числа — куски одной кривой, и «согласие с Ньютоном»
есть следствие выбора окна. Предсказание тут проверяется ФОРМОЙ, а не
значением: у него один параметр на девять полос.

КОНТРОЛИ:
  1. чистая степень `r^p` обязана дать местный показатель ПОСТОЯННЫМ —
     проверяется на искусственном профиле;
  2. подгонка одного `t` по девяти полосам: `chi²` и остатки обязаны
     быть без тренда, иначе форма не та;
  3. показатель, подогнанный по ОКНАМ из архива, обязан воспроизвести
     −0.5 и −1.06 из той же кривой.

  python3 deficit.py
"""
import glob
import math
import pickle

import numpy as np

from pencil import SCRATCH


def load():
    out = []
    for f in sorted(glob.glob(f'{SCRATCH}/bigfall_*.pkl')):
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
        out.append((r['n'], np.asarray(r['r_ell'], float),
                    np.asarray(r['per_seed'], float), r['rad'], r['ell']))
    return out


def local_exp(r, y, w=3):
    """Местный показатель в скользящем окне из w полос."""
    out = []
    for i in range(len(r) - w + 1):
        rr, yy = r[i:i + w], y[i:i + w]
        yy = np.abs(yy)
        m = yy > 0
        if m.sum() < 2:
            out.append((rr.mean(), float('nan'), float('nan'))); continue
        p = np.polyfit(np.log(rr[m]), np.log(yy[m]), 1)
        out.append((rr.mean(), p[0], 0.0))
    return np.array(out)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('КОНТРОЛЬ 1: у чистой степени местный показатель постоянен')
    rr = np.linspace(3, 11, 9)
    for p in (-0.5, -1.0):
        le = local_exp(rr, 10.0 * rr ** p)
        print(f'  r^{p}: местный {np.round(le[:,1],4)} '
              f'-> разброс {le[:,1].std():.2e}')
    print()
    data = load()
    print('ДАННЫЕ')
    for n, r, ps, rad, ell in data:
        print(f'  N = {n}, полос {len(r)}, засевов {ps.shape[1]}, '
              f'r от {r.min():.2f} до {r.max():.2f} l, '
              f'радиус дырки {rad/ell:.2f} l')
    print()
    # объединяем по N
    print('=' * 76)
    print('МЕСТНЫЙ ПОКАЗАТЕЛЬ ПО РАДИУСУ (окно 3 полосы)')
    for N in sorted({d[0] for d in data}):
        sel = [d for d in data if d[0] == N]
        r = sel[0][1]
        Y = np.hstack([d[2] for d in sel])
        y = np.abs(Y.mean(axis=1))
        le = local_exp(r, y)
        print(f'  N = {N}, миров {Y.shape[1]}:')
        print(f"{'r/l':>8}{'дефицит':>12}{'местный показ.':>16}")
        for i, (rc, p, _) in enumerate(le):
            print(f'{rc:>8.2f}{y[i+1]:>12.5f}{p:>16.3f}')
        print()
    print('=' * 76)
    print('ПОДГОНКА ФОРМЫ theory-doc: |Delta| = a/(t+r), ОДИН параметр t')
    print(f"{'N':>8}{'миров':>7}{'t/l':>9}{'chi²/полос':>13}"
          f"{'тренд остатков':>17}{'местный при r=t':>17}")
    for N in sorted({d[0] for d in data}):
        sel = [d for d in data if d[0] == N]
        r = sel[0][1]
        Y = np.hstack([d[2] for d in sel])
        y = np.abs(Y.mean(axis=1))
        e = Y.std(axis=1, ddof=1) / math.sqrt(Y.shape[1])
        best = None
        for t in np.arange(0.10, 200.0, 0.05):   # широкий скан: упор в край
            # был бы подписью негодной формы, а не ответом
            mdl = 1.0 / (t + r)
            a = ((y * mdl) / e ** 2).sum() / ((mdl * mdl) / e ** 2).sum()
            chi = (((y - a * mdl) / e) ** 2).sum()
            if best is None or chi < best[0]:
                best = (chi, t, a)
        chi, t, a = best
        res = (y - a / (t + r)) / e
        tr = np.polyfit(np.log(r), res, 1)[0]
        edge = 'УПОР В КРАЙ' if t < 0.2 or t > 199.0 else ''
        print(f'{N:>8}{Y.shape[1]:>7}{t:>9.2f}{chi/len(r):>13.2f}'
              f'{tr:>17.3f}{t:>17.2f}  {edge}')
    print()
    print('=' * 76)
    print('КОНТРОЛЬ 3: показатель по АРХИВНЫМ окнам из той же кривой')
    print(f"{'N':>8}{'окно':>22}{'показатель':>13}{'от -1':>9}")
    for N in sorted({d[0] for d in data}):
        sel = [d for d in data if d[0] == N]
        r = sel[0][1]
        Y = np.hstack([d[2] for d in sel])
        y = np.abs(Y.mean(axis=1))
        e = Y.std(axis=1, ddof=1) / math.sqrt(Y.shape[1])
        for nm, lo, hi in (('внутреннее 2.5-5.0 l', 2.5, 5.0),
                           ('внешнее r > 4.4 l', 4.4, 99.0),
                           ('всё вместе', 0.0, 99.0)):
            m = (r >= lo) & (r <= hi) & (y > 0)
            if m.sum() < 3:
                continue
            w = (y[m] / e[m]) ** 2
            p, cov = np.polyfit(np.log(r[m]), np.log(y[m]), 1,
                                w=np.sqrt(w), cov=True)
            sp = math.sqrt(cov[0, 0])
            print(f'{N:>8}{nm:>22}{f"{p[0]:+.3f}+-{sp:.3f}":>13}'
                  f'{abs(p[0]+1)/max(sp,1e-9):>9.1f}')
    print()
    print('''ЧТЕНИЕ. Если местный показатель монотонно крутеет, а форма
`a/(t+r)` садится с одним параметром без тренда остатков — профиль не
степенной, и оба архивных числа суть куски одной кривой. Тогда
«согласие с ньютоновским −1» есть следствие выбора окна, и это надо
записать вместо прежней формулировки.''')


if __name__ == '__main__':
    main()
