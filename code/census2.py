#!/usr/bin/env python3
"""ПЕРЕПИСЬ, дверь 1, со СОСТЯЗАТЕЛЬНЫМ поиском соперников.

census1.py провалил якорный контроль: архив (`actionexist-doc.md`)
доказал, что восемь объёмных счётов НЕ разделяют, а стенд на сетке из
690 секторов сказал «разделяют». Архив эту болезнь уже ловил один раз:
«разреженная сетка соперников дала ложную выполнимость; плотная (426
секторов) дала невыполнимость». Сетка — слабый способ: она не знает,
куда смотреть.

ЗДЕСЬ соперники генерируются ПРОТИВ текущего ответа:
  1. решаем ЛП, получаем theta;
  2. ищем сектор, минимизирующий theta·g - c, то есть сильнее всех
     нарушающий найденное действие (тонкая сетка по k и q плюс
     случайные профили размеров слоёв);
  3. добавляем нарушителей, повторяем.

Это срезающие плоскости. Если процесс сходится к невыполнимости с
действительным сертификатом Фаркаша — якорь воспроизведён, и стенду
можно верить. Если нарушителей не находится при тонком поиске —
разделимость настоящая, и тогда опровергнут архив.

Порядок важен: пока якорь не воспроизведён, расширенные наборы счётов
не считаются вовсе.
"""
import math

import numpy as np
from scipy.optimize import linprog

from census1 import EDGES, closure, feats, sprinkle_rc


def layered_S(n, sizes, q, rs):
    off = np.cumsum([0] + list(sizes))
    Rc = np.zeros((n, n), bool)
    for i in range(len(sizes) - 1):
        Rc[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = (
            rs.random((sizes[i], sizes[i + 1])) < q)
    h2 = (-q * math.log(q) - (1 - q) * math.log1p(-q)) if 0 < q < 1 else 0.
    S = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1)) * h2
    return closure(Rc), S


def profiles(n, k, rs, kind):
    if kind == 0:
        s = [n // k] * k
        s[-1] += n - sum(s)
    elif kind == 1:
        thin = max(1, n // (8 * k))
        s = [n // k] * k
        s[0] = s[-1] = thin
        s[k // 2] += n - sum(s)
    else:
        w = rs.random(k) ** 2 + 0.05
        s = np.maximum(1, np.rint(w / w.sum() * n).astype(int)).tolist()
        s[int(np.argmax(s))] += n - sum(s)
    return s if min(s) >= 1 and sum(s) == n else None


def make(n, k, q, kind, rs):
    s = profiles(n, k, rs, kind)
    if s is None:
        return None
    Rc, S = layered_S(n, s, q, rs)
    o, ab, b, R = feats(Rc, n)
    return o, ab, b, S, (k, round(q, 3), kind)


def solve(G, C):
    k = G.shape[1]
    A = np.hstack([-G, np.ones((len(G), 1))])
    r = linprog(np.concatenate([np.zeros(k), [-1.0]]), A_ub=A, b_ub=-C,
                bounds=[(None, None)] * (k + 1), method='highs')
    if r.status == 3:
        r2 = linprog(np.zeros(k), A_ub=-G, b_ub=-np.ones(len(G)),
                     bounds=[(None, None)] * k, method='highs')
        return math.inf, (r2.x if r2.status == 0 else np.ones(k))
    if r.status != 0:
        return None, None
    return float(r.x[-1]), r.x[:k]


def certify(G, C):
    m = len(G)
    r = linprog(-C, A_eq=G.T, b_eq=np.zeros(G.shape[1]),
                A_ub=np.ones((1, m)), b_ub=[1.0],
                bounds=[(0, None)] * m, method='highs')
    if r.status != 0:
        return None
    lam = r.x
    return float(np.abs(G.T @ lam).max()), float(C @ lam), lam


def verdict(t):
    if t == math.inf:
        return 'РАЗДЕЛЯЕТ (запас не ограничен)'
    if t > 0:
        return f'РАЗДЕЛЯЕТ, запас {t:+.4g}'
    return f'НЕ РАЗДЕЛЯЕТ (лучший запас {t:+.4g})'


def run(name, pick, f4, pool, S4, n, rs, rounds=10, per=80):
    """Срезающие плоскости. Цикл идёт, ПОКА действие существует и
    находятся нарушители. Отрицательный запас = действия нет, и
    добавление соперников может только ухудшить (устойчивая сторона)."""
    print(f'\n=== {name} ===')
    cur = list(pool)
    for it in range(rounds):
        Fj = np.array([pick(r) for r in cur], float)
        Sj = np.array([r[3] for r in cur])
        sc = np.maximum(np.abs(f4), 1.0)
        G = (f4[None, :] - Fj) / sc
        C = Sj - S4
        t, th = solve(G, C)
        if t is None:
            print('  ЛП не решилась')
            return None
        if t <= 0:
            print(f'  круг {it}: соперников {len(cur)} -> {verdict(t)}')
            c = certify(G, C)
            if c:
                res, sl, lam = c
                ok = res < 1e-6 and sl > 0
                print(f'  Фаркаш: |Gᵗλ| = {res:.2e}, Σλ·C = {sl:+.3f}'
                      f'  =>  сертификат '
                      f'{"ДЕЙСТВИТЕЛЕН" if ok else "НЕ действителен"}')
                top = np.argsort(lam)[::-1][:3]
                print('  смесь, в которую усредняется засев: ' + '; '.join(
                    f'λ={lam[i]:.3f} (k,q,вид)={cur[i][4]}'
                    for i in top if lam[i] > 1e-4))
            return t
        found, worst = 0, []
        for _ in range(per):
            k = int(rs.integers(2, min(n // 2, 320)))
            q = float(10 ** rs.uniform(-2.2, -0.004))
            m = make(n, k, q, int(rs.integers(0, 3)), rs)
            if m is None:
                continue
            g = (f4 - np.array(pick(m), float)) / sc
            val = float(th @ g) - (m[3] - S4)
            if val <= 0:
                found += 1
                worst.append((val, m))
        worst.sort(key=lambda z: z[0])
        print(f'  круг {it}: соперников {len(cur)}, {verdict(t)}, '
              f'нарушителей {found}/{per}', flush=True)
        if found == 0:
            print('  нарушителей нет при тонком состязательном поиске '
                  '=> разделимость держится')
            return t
        cur += [m for _, m in worst[:16]]
    print('  круги исчерпаны, разделимость ещё держится '
          '(вывод предварительный)')
    return t


def main():
    print(__doc__)
    n, rs = 400, np.random.default_rng(20261010)
    S4 = 4.0 * n * math.log(n)
    o4, ab4, b4, R4 = feats(sprinkle_rc(n, rs), n)
    pool = []
    for k in (2, 3, 4, 6, 9, 14, 22, 34, 55, 90, 140, 220):
        for q in (0.03, 0.08, 0.2, 0.45, 0.75, 0.95):
            for kind in (0, 1):
                m = make(n, k, q, kind, rs)
                if m:
                    pool.append(m)
    print(f'начальный пул: {len(pool)}; N = {n}, '
          f'энтропия засева 4N lnN = {S4:.0f}')
    sets = [
        ('ЯКОРЬ 1: восемь объёмных (архив: НЕ РАЗДЕЛЯЕТ)',
         lambda r: r[0], o4),
        ('ЯКОРЬ 2: + граничный счёт (архив: РАЗДЕЛЯЕТ)',
         lambda r: np.concatenate([r[0], r[2]]), np.concatenate([o4, b4])),
        ('НОВОЕ: полное распределение интервалов по лог-полосам',
         lambda r: r[1], ab4),
        ('НОВОЕ: распределение + восемь объёмных',
         lambda r: np.concatenate([r[1], r[0]]),
         np.concatenate([ab4, o4])),
    ]
    out = {}
    for nm, pick, f4 in sets:
        out[nm] = run(nm, pick, f4, pool, S4, n, rs)
    print('\n=== СВОДКА ===')
    for nm, t in out.items():
        print(f'  {nm.split(":")[0]:>8}: {verdict(t) if t is not None else "—"}'
              f'   [{nm.split(": ")[1]}]')


if __name__ == '__main__':
    main()
