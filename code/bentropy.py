#!/usr/bin/env python3
"""ДОСТАТОЧНОСТЬ ГРАНИЦЫ, мера без привязки к доле незаконных.

Прошлый замер (`compose-doc.md`) мерил ДОЛЮ СМЕШАННЫХ ГРУПП, и она
выросла 0.029 -> 0.111 при |A| 5->6. Но доля НЕЗАКОННЫХ шагов выросла
при этом в 6.4 раза (0.005 -> 0.032), то есть фон двигался, и вывода
сделать нельзя.

Мера без этого изъяна — ДОЛЯ ОБЪЯСНЁННОЙ ЭНТРОПИИ:

    f = 1 - H(законность | состояние) / H(законность)

Она нормирована на саму энтропию законности, значит от доли незаконных
не зависит по построению. f = 1 — состояние определяет законность
полностью; f = 0 — не знает о ней ничего.

СВОЯ ЛОВУШКА, и без неё мерить нельзя. H(·|состояние) падает САМА от
дробления: если групп столько же, сколько случаев, f = 1 при любом
бессмысленном состоянии. Поэтому считается ПОЛ: состояния
перемешиваются между случаями с сохранением размеров групп, и f
считается заново. Истинный сигнал — ПРЕВЫШЕНИЕ над полом.

Это ровно тот урок, что за сегодня попался четырежды: мера обязана
иметь пол, не зависящий от моего выбора.

КОНТРОЛЬ: S5 (весь блок) обязан дать f = 1 ровно.
"""
import math
import os
import pickle

import numpy as np

from boundary import states
from compose2 import closure, dim_le_2, posets


def H(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def explained(keys, labels, rs, nshuf=40):
    """f и его пол из перемешивания с теми же размерами групп."""
    lab = np.array(labels, bool)
    n = len(lab)
    Hy = H(float(lab.mean()))
    if Hy == 0:
        return float('nan'), float('nan'), 0
    g = {}
    for i, k in enumerate(keys):
        g.setdefault(k, []).append(i)
    sizes = [len(v) for v in g.values()]

    def cond(assign):
        tot = 0.0
        for idxs in assign:
            m = lab[idxs].mean()
            tot += len(idxs) / n * H(float(m))
        return tot
    f = 1.0 - cond(list(g.values())) / Hy
    floors = []
    for _ in range(nshuf):
        perm = rs.permutation(n)
        pos, parts = 0, []
        for s in sizes:
            parts.append(perm[pos:pos + s])
            pos += s
        floors.append(1.0 - cond(parts) / Hy)
    return f, float(np.mean(floors)), len(g)


def main():
    print(__doc__)
    rs = np.random.default_rng(10102026)
    cache = '/tmp/claude-0/posets.pkl'
    db = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    res = {}
    for a in (4, 5, 6):
        if a not in db:
            db[a] = posets(a)
            pickle.dump(db, open(cache, 'wb'))
        PA = [R for R in db[a] if dim_le_2(R)]
        rows, labels = [], []
        for RA in PA:
            for bits in range(1 << a):
                D = np.array([(bits >> i) & 1 for i in range(a)], bool)
                if (RA & D[None, :] & ~D[:, None]).any():
                    continue
                M = np.zeros((a + 1, a + 1), bool)
                M[:a, :a] = RA
                M[:a, a] = D
                labels.append(dim_le_2(closure(M)))
                st = states(RA, D)
                rows.append(st if a <= 5 else
                            {k: v for k, v in st.items() if k != 'S5'})
        lab = np.array(labels, bool)
        print(f'\n|A| = {a}: шагов {len(lab)}, законных '
              f'{lab.mean():.4f}, H(законность) = '
              f'{H(float(lab.mean())):.4f} бит')
        print(f"{'состояние':>11}{'групп':>7}{'f':>9}{'пол':>9}"
              f"{'превышение':>13}")
        res[a] = {}
        for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
            if k not in rows[0]:
                continue
            f, fl, ng = explained([r[k] for r in rows], lab, rs)
            res[a][k] = (f, fl, f - fl)
            print(f'{k:>11}{ng:>7}{f:>9.4f}{fl:>9.4f}{f-fl:>13.4f}'
                  + ('   <-- ДОСТАТОЧНО' if f > 0.9999 else ''))

    print('\nРОСТ ПО РАЗМЕРУ БЛОКА (главный вопрос)')
    print(f"{'состояние':>11}" + ''.join(f'{f"|A|={a}":>12}' for a in res)
          + '   чтение')
    for k in ('S1', 'S2', 'S3', 'S4'):
        vals = [res[a][k][2] for a in res if k in res[a]]
        trend = ('растёт' if vals[-1] > vals[0] + 0.02 else
                 'падает' if vals[-1] < vals[0] - 0.02 else 'ровно')
        print(f'{k:>11}' + ''.join(f'{v:>12.4f}' for v in vals)
              + f'   превышение {trend}')
    print('\n  (превышение над полом — это и есть сигнал; падение '
          'означает,')
    print('   что граница с ростом блока объясняет законность хуже)')


if __name__ == '__main__':
    main()
