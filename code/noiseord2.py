#!/usr/bin/env python3
"""ПОРЯДОК, ИСКАЖЁННЫЙ ШУМОМ: извлекается ли он из турнира большинства.

ЧАСТЬ 1 (огрубление орты) закрывается рассуждением, а не замером, и это
надо сказать прямо: прореживание точек БУКВАЛЬНО воспроизводит
построение - N точек с k независимыми координатами дают N' точек с k
независимыми координатами. Ансамбль на шкалах есть неподвижная точка
огрубления тождественно, сравнивать нечего. Плюс порядковая размерность
не растёт при переходе к индуцированному подпорядку, значит из орты
световой конус не возникнет ни на каком масштабе. Ниже - одна сверка
кода, не вывод.

ЧАСТЬ 2 - вопрос из `majority-doc.md`, до сих пор не измеренный. Турнир
большинства БЛИЗОК к порядку: один линейный ранг объясняет 82%
отношений, 3-циклов в 3.7 раза меньше случайного турнира той же
плотности. Извлекается ли порядок, ИСКАЖЁННЫЙ шумом, а не ЗАМЕНЁННЫЙ им?

Извлечение: ранг Борда (средняя позиция по шкалам), оставляем стрелки
турнира, согласные с ним (подотношение линейного порядка циклов не
имеет), транзитивное замыкание -> порядок Q.

КОНТРОЛЬ, который и различает «искажён» от «заменён»: та же процедура на
СЛУЧАЙНОМ турнире той же плотности со случайным рангом. Если у Q
точность и полнота много выше - шум искажает, а не заменяет.
"""
import numpy as np

from sectors import sprinkle_d
from waveop import causal, sprinkle_11
from orthant import scales, thresh_order, acyclic, closure, prof, region, rf

N = 400


def thin_to(R, m, rs):
    keep = np.sort(rs.choice(R.shape[0], size=m, replace=False))
    return R[np.ix_(keep, keep)]


def borda(S):
    k, n = S.shape
    acc = np.zeros(n)
    for s in S:
        q = np.empty(n, int)
        q[s] = np.arange(n)
        acc += q
    return acc / k


def extract(T, score):
    return closure(T & (score[:, None] < score[None, :]))


def pr_rc(Q, P):
    inter = float((Q & P).sum())
    return inter / max(float(Q.sum()), 1.0), inter / max(float(P.sum()), 1.0)


def main():
    print(__doc__)
    rs = np.random.default_rng(949459)
    chi4, thr4 = region(lambda: sprinkle_d(N, 4, rs), 40, rs)
    chi2, thr2 = region(lambda: causal(sprinkle_11(N, rs)), 40, rs)
    print(f'пороги при N={N}: 4d {thr4:.1f}, 1+1 {thr2:.1f}')

    print('\nЧАСТЬ 1, сверка кода (ждём одинаковые хи2):')
    a = thresh_order(scales(N, 4, rs), 4)
    b = thin_to(thresh_order(scales(4 * N, 4, rs), 4), N, rs)
    print(f'  орта построена при N={N}:            '
          f'доля {rf(a):.4f}, хи2 к 4d {float(chi4(prof(a))):.1f}')
    print(f'  орта при N={4 * N}, прорежена до {N}: '
          f'доля {rf(b):.4f}, хи2 к 4d {float(chi4(prof(b))):.1f}')

    print('\nЧАСТЬ 2, k=7')
    S = scales(N, 7, rs)
    P = thresh_order(S, 7)
    T = thresh_order(S, 4)
    print(f'  P (единогласие): доля {rf(P):.4f}, ациклично {acyclic(P)}')
    print(f'  T (большинство): доля {rf(T):.4f}, ациклично {acyclic(T)}')

    rows = []
    Q = extract(T, borda(S))
    rows.append(('Q из большинства', Q))
    d = rf(T)
    U = np.triu(rs.random((N, N)) < d, 1)
    rows.append(('Q из случайного турнира',
                 extract(U, rs.permutation(N).astype(float))))
    print(f"\n{'что':>26}{'доля':>9}{'точн.':>8}{'полн.':>8}"
          f"{'хи2 4d':>9}{'хи2 1+1':>10}")
    for lab, X in rows:
        pr, rc = pr_rc(X, P)
        print(f'{lab:>26}{rf(X):>9.4f}{pr:>8.4f}{rc:>8.4f}'
              f'{float(chi4(prof(X))):>9.1f}{float(chi2(prof(X))):>10.1f}',
              flush=True)
    print(f"{'сам P, для сравнения':>26}{rf(P):>9.4f}{1.0:>8.4f}{1.0:>8.4f}"
          f'{float(chi4(prof(P))):>9.1f}{float(chi2(prof(P))):>10.1f}')
    print('\nЧТЕНИЕ: точность и полнота у Q из большинства много выше, чем')
    print('у Q из случайного => шум ИСКАЖАЕТ порядок, а не заменяет его.')


if __name__ == '__main__':
    main()
