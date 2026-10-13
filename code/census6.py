#!/usr/bin/env python3
"""ПЕРЕПИСЬ, решающее число: как растёт ДЕФИЦИТ ЭНТРОПИИ с N.

Все счётные двери закрыты (census3-5). Остаётся дверь ВОГНУТОСТИ, и у
неё есть теорема в основании: строго вогнутая функция МОЖЕТ иметь
строгий максимум во внутренней точке выпуклой оболочки, а линейная не
может. Значит вопрос не «бывает ли», а «хватит ли величины».

Хватит, если вогнутая добавка F даёт разрыв по Йенсену

    F(f_засев) - Σ λ_j F(f_j)  >  Σ λ_j S_j - S_засев  =  Σ λ·C

Правая часть — дефицит энтропии в состязательном оптимуме. При N=400
он вышел 46.69 нат против энтропий ~10^4. ВСЁ решает его рост:
  * растёт как N^2 — ни одна добавка с ограниченным вкладом на элемент
    не победит, дверь закрыта;
  * растёт как N lnN или медленнее — добавка порядка N побеждает.

ПРОТОКОЛ ОДИН И ТОТ ЖЕ при каждом N (иначе числа несравнимы):
начальная сетка, 6 состязательных кругов по 60 выборок, затем Σλ·C в
полученном минимаксе. Протокол фиксирован ДО прогона.

КОНТРОЛЬ: на каждом N проверяется, что сертификат Фаркаша
действителен (|Gᵗλ| < 1e-6 и Σλ·C > 0) — без него число не имеет
смысла.
"""
import math

import numpy as np

from census1 import feats, sprinkle_rc
from census2 import certify, make, solve, verdict
from census3 import perc, sprink_d

ROUNDS, PER, ADD = 6, 60, 14


def deficit(n, rs):
    S4 = 4.0 * n * math.log(n)
    o4, ab4, b4, _ = feats(sprinkle_rc(n, rs), n)
    f4 = np.concatenate([ab4, b4])
    sc = np.maximum(np.abs(f4), 1.0)

    def vec(m):
        return np.concatenate([m[1], m[2]])

    pool = []
    for k in (2, 3, 4, 6, 10, 18, 32, 60, max(2, n // 4)):
        for q in (0.03, 0.08, 0.2, 0.45, 0.8):
            for kind in (0, 1, 2):
                m = make(n, k, q, kind, rs)
                if m:
                    pool.append(m)
    for p in (0.006, 0.02, 0.06, 0.2):
        pool.append(perc(n, p, rs))
    for d in (2, 3, 5):
        pool.append(sprink_d(n, d, rs, S4))

    th = None
    for it in range(ROUNDS):
        G = np.array([(f4 - vec(m)) / sc for m in pool])
        C = np.array([m[3] for m in pool]) - S4
        t, th = solve(G, C)
        if t is None or t <= 0:
            break
        add = []
        for _ in range(PER):
            k = int(rs.integers(2, max(3, n // 2)))
            q = float(10 ** rs.uniform(-2.3, -0.004))
            m = make(n, k, q, int(rs.integers(0, 3)), rs)
            if m is None:
                continue
            if float(th @ ((f4 - vec(m)) / sc)) - (m[3] - S4) <= 0:
                add.append(m)
        if not add:
            return None, t, len(pool), None
        pool += add[:ADD]
    G = np.array([(f4 - vec(m)) / sc for m in pool])
    C = np.array([m[3] for m in pool]) - S4
    t, _ = solve(G, C)
    c = certify(G, C)
    if c is None:
        return None, t, len(pool), None
    res, sl, lam = c
    ok = res < 1e-6 and sl > 0
    return (sl if ok else None), t, len(pool), res


def main():
    print(__doc__)
    rs = np.random.default_rng(11235813)
    print(f"{'N':>6}{'соперников':>12}{'запас':>12}{'дефицит Σλ·C':>15}"
          f"{'|Gᵗλ|':>10}{'дефицит/N':>12}{'/(N lnN)':>11}{'/N²':>11}")
    Ns, D = [], []
    for n in (150, 220, 300, 420, 600):
        sl, t, m, res = deficit(n, rs)
        if sl is None:
            print(f'{n:>6}{m:>12}{verdict(t):>12}'
                  f'{"— сертификата нет":>15}')
            continue
        Ns.append(n); D.append(sl)
        print(f'{n:>6}{m:>12}{t:>12.4g}{sl:>15.3f}{res:>10.0e}'
              f'{sl/n:>12.4f}{sl/(n*math.log(n)):>11.5f}'
              f'{sl/n**2:>11.6f}', flush=True)
    if len(Ns) >= 3:
        Ns = np.array(Ns, float); D = np.array(D)
        p = np.polyfit(np.log(Ns), np.log(D), 1)
        print(f'\n  дефицит ~ N^{p[0]:.3f}')
        print('  местные показатели: ' + ', '.join(
            f'{int(Ns[i])}->{int(Ns[i+1])}: '
            f'{math.log(D[i+1]/D[i])/math.log(Ns[i+1]/Ns[i]):.3f}'
            for i in range(len(Ns) - 1)))
        print(f'\n  ВЕРДИКТ: показатель {p[0]:.2f} — '
              + ('ближе к 2, дверь вогнутости закрыта так же, как '
                 'линейная' if p[0] > 1.6 else
                 'заметно меньше 2, значит добавка порядка N или '
                 'N lnN может победить — дверь открыта'))


if __name__ == '__main__':
    main()
