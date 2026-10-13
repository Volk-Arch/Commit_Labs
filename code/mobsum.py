#!/usr/bin/env python3
"""ТОЧНАЯ КОГЕРЕНТНАЯ СУММА с действием Мёбиуса: гибнет ли сектор.

mobius.py показал: функция Мёбиуса различает секторы, причём в нужную
сторону — у засева sum mu гасится внутри конфигурации (2.8% остатка), у
слоистых не гасится (38...92%), и по величине слоистое действие больше
в ~170 раз. Контроли там прошли точно (Z@M - I = 0).

Здесь считается то, что решает: ТОЧНАЯ когерентная сумма
|sum_конфигурации exp(i*th*S_mu)| против счёта сектора. Гауссова оценка
НЕ используется — вчера она подвела именно на этом секторе
(`cancel5-doc.md`: гаусс предсказывал гибель, точная сумма дала рост).

ЗАМКНУТЫЙ ВИД для трёх слоёв A(a) - B(b) - C(c). Соседние пары не имеют
промежуточных, значит все они связи с mu = -1. Интервал между x из A и
z из C есть антицепь {y : P[x,y] и Q[y,z]}, значит mu = m - 1. Отсюда

    S_mu = -|P| - |Q| + sum_y d_вх(y) d_вых(y) - R_AC

где R_AC — число связанных пар A->C. Квадратичная форма по степеням —
ровно то, чего нет ни в одном аффинном по счётам действии.

КОНТРОЛЬ A: замкнутый вид сверяется с прямым счётом mu через обращение
зета-матрицы на случайных конфигурациях.
КОНТРОЛЬ B: действие БД на тех же конфигурациях обязано дать
ВЫЖИВАНИЕ сектора (архив: c > 0 всюду) — если и оно даст гибель, значит
сломан стенд.

ЧТЕНИЕ: сектор гибнет, если ln|сумма| растёт медленнее энтропии
bits*ln2, то есть доля выжившей энтропии -> 0.
"""
import itertools
import math

import numpy as np

from mobius import mobius

K4 = 4.0 / math.sqrt(6.0)


def all_configs(a, b, c):
    """Все матрицы P (a x b) и Q (b x c) как битовые маски."""
    nb_p, nb_q = a * b, b * c
    P = ((np.arange(1 << nb_p)[:, None] >> np.arange(nb_p)) & 1
         ).reshape(-1, a, b).astype(np.int8)
    Q = ((np.arange(1 << nb_q)[:, None] >> np.arange(nb_q)) & 1
         ).reshape(-1, b, c).astype(np.int8)
    return P, Q


def s_mu_closed(P, Q):
    """S_mu для всех пар (P, Q) — векторно."""
    M = np.einsum('iab,jbc->ijac', P, Q)        # (nP, nQ, a, c)
    din = P.sum(axis=1)                          # (nP, b)
    dout = Q.sum(axis=2)                         # (nQ, b)
    quad = din @ dout.T                          # (nP, nQ)
    return (-P.sum(axis=(1, 2))[:, None] - Q.sum(axis=(1, 2))[None, :]
            + quad - (M > 0).sum(axis=(2, 3)))


def s_bd_closed(P, Q, n):
    """S_БД/k4 на тех же конфигурациях: N - N0 + 9N1 - 16N2 + 8N3."""
    M = np.einsum('iab,jbc->ijac', P, Q)
    N0 = (P.sum(axis=(1, 2))[:, None] + Q.sum(axis=(1, 2))[None, :]
          + (M == 0).sum(axis=(2, 3)) * 0)
    out = n - N0.astype(float)
    for m, w in ((1, 9.0), (2, -16.0), (3, 8.0)):
        out = out + w * (M == m).sum(axis=(2, 3))
    return out


def build(a, b, c, p, q):
    n = a + b + c
    R = np.zeros((n, n), bool)
    R[:a, a:a + b] = p.astype(bool)
    R[a:a + b, a + b:] = q.astype(bool)
    C = R | ((R.astype(np.int8) @ R.astype(np.int8)) > 0)
    return C


def main():
    print(__doc__)
    rs = np.random.default_rng(2718)

    print('КОНТРОЛЬ A: замкнутый вид против обращения зета-матрицы')
    bad = 0
    for _ in range(25):
        a, b, c = 3, 3, 3
        p = (rs.random((a, b)) < 0.5).astype(np.int8)
        q = (rs.random((b, c)) < 0.5).astype(np.int8)
        Rc = build(a, b, c, p, q)
        R, Mm = mobius(Rc)
        direct = int(Mm[R].sum())
        closed = int(s_mu_closed(p[None], q[None])[0, 0])
        if direct != closed:
            bad += 1
            print(f'  расхождение: прямо {direct}, замкнуто {closed}')
    print(f'  {25-bad}/25 совпали точно'
          f'{"" if bad == 0 else "   <-- СТЕНД СЛОМАН"}')
    if bad:
        return

    ths = np.linspace(0.01, math.pi, 400)
    print(f"\n{'слои':>10}{'бит':>5}{'энтропия':>10}"
          f"{'действие':>10}{'ln|сумма|':>11}{'доля выжив.':>13}{'th':>7}")
    for (a, b, c) in ((2, 2, 2), (3, 3, 3), (2, 4, 2), (3, 4, 3)):
        P, Q = all_configs(a, b, c)
        bits = a * b + b * c
        ent = bits * math.log(2)
        for nm, S in (('Мёбиус', s_mu_closed(P, Q)),
                      ('БД', s_bd_closed(P, Q, a + b + c))):
            Sf = S.ravel().astype(float)
            best = (np.inf, 0.0)
            for th in ths:
                z = abs(np.exp(1j * th * Sf).sum())
                l = math.log(max(z, 1e-300))
                if l < best[0]:
                    best = (l, th)
            print(f'{f"{a}-{b}-{c}":>10}{bits:>5}{ent:>10.2f}'
                  f'{nm:>10}{best[0]:>11.3f}{best[0]/ent:>13.4f}'
                  f'{best[1]:>7.3f}', flush=True)

    print('\nЧТЕНИЕ: доля выжившей энтропии ln|сумма| / (bits*ln2).')
    print('У БД она обязана быть заметно положительной (архив: сектор')
    print('выживает всюду). Если у Мёбиуса она много меньше или')
    print('отрицательна — знак от инварианта формы делает то, чего не')
    print('смог ни один вес на счётах.')


if __name__ == '__main__':
    main()
