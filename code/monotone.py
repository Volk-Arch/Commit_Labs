#!/usr/bin/env python3
"""ВСЕ ПРАВИЛА НА ШКАЛАХ: только единогласие на подмножестве.

Последняя незакрытая лазейка: у орты один целый параметр k, и крона её
поймала (`crown-doc.md`) именно потому, что ручка одна. Но правила с
ВЕСАМИ шкал в проекте не проверялись ни разу - а у них ручек больше.

Утверждение (теорема, не замер). Правило есть монотонная булева функция
f от битов «раньше на шкале s». Пусть i<j с набором битов a и j<l с
набором b. Про i<l гарантированно известны биты только там, где ОБА
единицы, то есть худший случай есть a AND b. Значит транзитивность при
ЛЮБОЙ конфигурации требует
    f(a) и f(b)  =>  f(a ∧ b),
а монотонное семейство, замкнутое относительно пересечения, есть в
точности ГЛАВНЫЙ ФИЛЬТР {x : x ⊇ S}. То есть единственные транзитивные
правила - единогласие на подмножестве шкал, и ВЕСА НЕ ДАЮТ НИЧЕГО
НОВОГО.

Это сильнее `majority-doc.md`, где проверялись лишь симметричные пороги
m из k.

ПРОВЕРКА исчерпывающая: перебираем ВСЕ монотонные функции от k=4 битов
(их 168 по числу Дедекинда), для каждой строим отношение на случайной
конфигурации и проверяем частичный порядок топологической сортировкой.
Обязаны пройти ровно главные фильтры.

КОНТРОЛЬ: число монотонных функций обязано выйти 168 (k=4) и 20 (k=3) -
числа Дедекинда. Если нет, перебор неверен.
"""
import numpy as np

DED = {2: 6, 3: 20, 4: 168}


def monotone_fns(k):
    n = 1 << k
    bits = [[(m >> s) & 1 for s in range(k)] for m in range(n)]
    sub = [[m2 for m2 in range(n) if (m2 & m) == m2] for m in range(n)]
    out = []
    for tt in range(1 << n):
        f = [(tt >> m) & 1 for m in range(n)]
        ok = True
        for m in range(n):
            if f[m]:
                continue
            for m2 in sub[m]:
                if f[m2]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(tuple(f))
    return out


def principal(f, k):
    """является ли f главным фильтром {x : x ⊇ S}."""
    n = 1 << k
    acc = [m for m in range(n) if f[m]]
    if not acc:
        return None
    S = acc[0]
    for m in acc:
        S &= m
    if all(f[m] == (1 if (m & S) == S else 0) for m in range(n)):
        return S
    return None


def relation(f, S, k, n):
    """R[i,j] = f(биты «i раньше j»)."""
    pos = []
    for s in range(k):
        q = np.empty(n, int)
        q[S[s]] = np.arange(n)
        pos.append(q)
    code = np.zeros((n, n), np.int64)
    for s in range(k):
        code |= ((pos[s][:, None] < pos[s][None, :]).astype(np.int64)
                 << s)
    fv = np.array(f, bool)
    R = fv[code]
    np.fill_diagonal(R, False)
    return R


def is_order(R):
    if (R & R.T).any():
        return False, 'не антисимметрично'
    n = R.shape[0]
    deg = R.sum(axis=0).astype(int)
    alive = np.ones(n, bool)
    left = n
    while left:
        z = np.nonzero(alive & (deg == 0))[0]
        if len(z) == 0:
            return False, 'цикл'
        alive[z] = False
        deg -= R[z].sum(axis=0).astype(int)
        deg[z] = -1
        left -= len(z)
    F = R.astype(np.float32)
    if ((F @ F > 0.5) & ~R).any():
        return False, 'не транзитивно'
    return True, 'порядок'


def main():
    print(__doc__)
    rs = np.random.default_rng(1618)
    for k in (3, 4):
        fns = monotone_fns(k)
        ok = len(fns) == DED[k]
        print(f'\nk={k}: монотонных функций {len(fns)}, '
              f'число Дедекинда {DED[k]} => {"ОК" if ok else "СБОЙ"}')
        n = 400
        Sc = np.array([rs.permutation(n) for _ in range(k)])
        good, bad, misclass = [], {}, []
        for f in fns:
            P = principal(f, k)
            R = relation(f, Sc, k, n)
            isord, why = is_order(R)
            if isord:
                good.append((f, P))
            else:
                bad[why] = bad.get(why, 0) + 1
            if isord != (P is not None):
                misclass.append((f, P, why))
        print(f'  дали частичный порядок: {len(good)} из {len(fns)}')
        print(f'  провалились: ' + ', '.join(f'{v} ({kk})'
                                             for kk, v in bad.items()))
        np_ = sum(1 for f in fns if principal(f, k) is not None)
        print(f'  главных фильтров среди них: {np_}')
        print(f'  расхождений «порядок ⟺ главный фильтр»: '
              f'{len(misclass)} => '
              f'{"ТЕОРЕМА ПОДТВЕРЖДЕНА" if not misclass else "СБОЙ"}')
        if misclass:
            for f, P, why in misclass[:3]:
                print(f'    {f} главный={P} причина={why}')


if __name__ == '__main__':
    main()
