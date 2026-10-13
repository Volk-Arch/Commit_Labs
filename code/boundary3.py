#!/usr/bin/env python3
"""ДОСТАТОЧНА ЛИ ГРАНИЦА: сколько состояния нужно для законности шага.

Гипотеза (наша; другой ИИ назвал её установленной): состояние, которое
блок передаёт будущему, несёт ГРАНИЦА. Против простой версии говорит
сегодняшний замер: ширина как ЧИСЛО воспроизводится смесью врага ровно
(182.0 против 182). Значит вопрос в СТРУКТУРЕ границы.

Поправка к доводу: координаты НЕ недоступны в порядке. Келли–Троттер
даёт лишь то, что двумерность не описывается КОНЕЧНЫМ списком
запретов; реализатор порядком определён. Нужны не лишние данные, а
нелокальное либо состоятельное чтение.

Постановка. Шаг роста: к блоку A добавляется элемент с прошлым D
(нижнее множество). Законность = результат ещё двумерен. Вопрос: по
какому объёму данных законность ОПРЕДЕЛЕНА?

ЛЕСТНИЦА СОСТОЯНИЙ:
  S1 — только число максимальных элементов D (аналог ширины-числа);
  S2 — счётный портрет разреза: |D|, |max D|, |A|;
  S3 — помеченный подпорядок на max(D) ∪ min(A\\D), то есть сам разрез;
  S4 — он же плюс max(A);
  S5 — весь A с пометкой D (по построению достаточен).

Для каждого уровня случаи группируются по значению состояния и ищется
группа, где есть И законный, И незаконный случай. Если есть —
состояние недостаточно. Получается не «да/нет», а ЧИСЛО: на каком
уровне смешивание исчезает, столько состояния и обязано течь.

КОНТРОЛЬ: S5 обязан дать НОЛЬ смешанных групп.
"""
import itertools
import os
import pickle

import numpy as np

from compose2 import closure, dim_le_2, posets


def canon_marked(M, marks):
    n = len(M)
    best = None
    mk = np.array(marks, np.uint8)
    for p in itertools.permutations(range(n)):
        q = np.array(p)
        key = (M[np.ix_(q, q)].astype(np.uint8).tobytes(),
               mk[q].tobytes())
        if best is None or key < best:
            best = key
    return best


def states(RA, D):
    a = len(RA)
    idx = np.arange(a)
    inD = D.astype(bool)
    maxD = [int(i) for i in idx[inD] if not (RA[i] & inD).any()]
    rest = idx[~inD]
    minR = [int(j) for j in rest if not (RA[:, j] & ~inD).any()]
    maxA = [int(i) for i in idx if not RA[i].any()]

    def sub(sel, marker):
        sel = sorted(set(sel))
        if not sel:
            return b''
        return canon_marked(RA[np.ix_(sel, sel)],
                            [1 if s in marker else 0 for s in sel])

    return {
        'S1': (len(maxD),),
        'S2': (int(inD.sum()), len(maxD), a),
        'S3': sub(maxD + minR, set(maxD)),
        'S4': sub(maxD + minR + maxA, set(maxD)),
        'S5': (canon_marked(RA, inD.astype(np.uint8))
               if a <= 5 else b'skip'),
    }


def main():
    print(__doc__)
    cache = '/tmp/claude-0/posets.pkl'
    db = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    for a in (5, 6):
        if a not in db:
            db[a] = posets(a)
            pickle.dump(db, open(cache, 'wb'))
        PA = [R for R in db[a] if dim_le_2(R)]
        groups = {k: {} for k in ('S1', 'S2', 'S3', 'S4', 'S5')}
        cases = legal_n = 0
        ex = {}
        for RA in PA:
            for bits in range(1 << a):
                D = np.array([(bits >> i) & 1 for i in range(a)], bool)
                if (RA & D[None, :] & ~D[:, None]).any():
                    continue
                M = np.zeros((a + 1, a + 1), bool)
                M[:a, :a] = RA
                M[:a, a] = D
                legal = dim_le_2(closure(M))
                cases += 1
                legal_n += int(legal)
                for k, v in states(RA, D).items():
                    groups[k].setdefault(v, {})
                    groups[k][v].setdefault(legal, (RA, D))
        print(f'\n|A| = {a}: блоков {len(PA)}, шагов {cases}, '
              f'законных {legal_n} ({legal_n/max(cases,1):.3f})')
        print(f"{'состояние':>11}{'групп':>8}{'смешанных':>11}{'доля':>8}")
        for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
            if a > 5 and k == 'S5':
                print(f'{k:>11}{"пропущен (контроль пройден при a=5)":>30}')
                continue
            g = groups[k]
            mixed = [v for v in g.values() if len(v) > 1]
            print(f'{k:>11}{len(g):>8}{len(mixed):>11}'
                  f'{len(mixed)/max(len(g),1):>8.3f}'
                  + ('   <-- ДОСТАТОЧНО' if not mixed else ''))
            if mixed and k == 'S3' and a == 6 and k not in ex:
                ex[k] = mixed[0]
        if a <= 5 and any(len(v) > 1 for v in groups['S5'].values()):
            print('  ВНИМАНИЕ: S5 смешивает — сломан перебор; '
                  'остальные строки не считаются')
        for k, v in ex.items():
            print(f'  пример смешивания на {k}:')
            for legal, (RA, D) in v.items():
                print(f'    законно={legal}, D={D.astype(int).tolist()}, '
                      f'A=\n{RA.astype(int)}')


if __name__ == '__main__':
    main()
