#!/usr/bin/env python3
"""ГДЕ СЭМПЛЕРУ МОЖНО ВЕРИТЬ: РАЗБРОС ПО ЗЁРНАМ ПРОТИВ БЛОЧНОЙ ОШИБКИ.

critmix.py нашёл настоящую причину: на прежней лестнице доли принятых
ОБМЕНОВ в критических парах 0.07/0.17/0.09 - темперирование не работало.
Густая лестница чинит обмены (минимум 0.19), но при N=24 равновесия всё
равно нет: половины расходятся на 4.7 сигмы, а H по независимым прогонам
гуляет 3.87...4.59 при заявленных ошибках 0.06-0.12.

ДВА УРОКА, оба записываю:
  * застрявшая цепь ПРОХОДИТ проверку половин (прежняя лестница дала
    2.0 сигмы, «равновесие есть») - значит половины необходимы, но не
    достаточны;
  * честная ошибка - разброс по НЕЗАВИСИМЫМ зёрнам, и он при N=24
    вчетверо больше блочной.

Поэтому: берём N, где разброс по зёрнам СОВПАДАЕТ с блочной ошибкой
(сэмплеру можно верить), и выносим вердикт о геометричности ТАМ, с
ошибкой из разброса, а не из блоков.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ: при честной ошибке отклонения либо остаются
значимыми (фаза негеометрична), либо исчезают (вердикта нет).
"""
import json
import math
import sys

import numpy as np

from critphase import summarize
from critmix import ladder, NEW
from critgeom3 import stats, best, box, KEYS
from crithalf import get_model

SEEDS = (4001, 4002, 4003)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    nsweep = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    print(__doc__)
    rs0 = np.random.default_rng(777000 + n)
    mod, dias, cached = get_model(n, rs0)
    m, e = stats(lambda: box(n, 2, 2.0, rs0), 120, rs0)
    c0, g0, _ = best(m, e, mod, dias)
    print(f'КОНТРОЛЬ d=2 a=2.0 -> {g0}, хи2={c0:.1f} '
          f'=> {"ОК" if c0 < 8 else "СБОЙ"}')

    k = NEW.index(0.04)
    per = []
    for sd in SEEDS:
        rs = np.random.default_rng(sd)
        recs, swr, mvr = ladder(n, NEW, nsweep, rs)
        tm, te = {}, {}
        for key in KEYS:
            mu, er = summarize(recs[k], key)
            tm[key], te[key] = float(mu), float(er)
        per.append((tm, te))
        print(f'  зерно {sd}: обмены мин {swr.min():.2f}, '
              f'ходы {mvr[k]:.3f},  '
              + '  '.join(f'{x}={tm[x]:.3f}±{te[x]:.3f}' for x in KEYS),
              flush=True)

    print(f'\nN={n}: разброс по зёрнам против блочной ошибки')
    tm, te = {}, {}
    for key in KEYS:
        vs = np.array([p[0][key] for p in per])
        blk = float(np.mean([p[1][key] for p in per]))
        sct = float(vs.std(ddof=1) / math.sqrt(len(vs)))
        tm[key] = float(vs.mean())
        te[key] = max(blk, sct)
        print(f'  {key}: среднее {vs.mean():.3f}, блочная {blk:.3f}, '
              f'по зёрнам {sct:.3f}, отношение {sct / blk:.2f}')
    print('  отношение около 1 => сэмплеру верим; много больше => нет')

    chi, geo, sig = best(tm, te, mod, dias)
    print(f'\nвердикт с ЧЕСТНОЙ ошибкой: хи2={chi:.1f}, {geo}')
    print('  отклонения: '
          + '  '.join(f'{x}={sig[x]:+.1f}σ' for x in KEYS))
    with open(f'/tmp/claude-0/critseeds-{n}.json', 'w') as f:
        json.dump(dict(n=n, nsweep=nsweep, chi=chi, geo=geo,
                       tm=tm, te=te, sig=sig), f)


if __name__ == '__main__':
    main()
