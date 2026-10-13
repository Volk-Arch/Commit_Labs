#!/usr/bin/env python3
"""ЕСЛИ СВЯЗИ РВУТСЯ: сколько их можно отрезать.

Возражение, и оно верное: бесконечного числа связей быть не может. В
бесконечном засеве у каждого события их бесконечно много - у прошлого
светового конуса бесконечный объём. Это известная проблема нелокальности
причинных множеств, и сегодняшняя стена шума (`noisewall-doc.md`,
sigma ~ N^1.001) есть её количественное лицо: разброс растёт ИМЕННО
потому, что оператор суммирует по неограниченному прошлому.

Прямая проверка альтернативы «связи рвутся»: берём настоящий засев 1+1,
ОБРЕЗАЕМ каждому элементу связи до k ближайших по собственному времени
(tau^2 = du*dv), остальное доделывает транзитивное замыкание. Если при
малом k мир не портится - лишние связи лишние, и возражение
конструктивно.

МЕРЫ (весь собранный сегодня фейсконтроль):
  * доля связанных (обязана быть 0.5 в 1+1);
  * доля связанных ВНУТРИ интервалов (обязана быть 0.5, `facecontrol-doc`);
  * число корон S_3 (обязано быть НОЛЬ, `crown-doc`: dim=2 ровно);
  * среднее число связей на элемент до и после.

КОНТРОЛЬ: при k = бесконечность обрезание ничего не меняет, и все четыре
величины обязаны совпасть с исходными.
"""
import numpy as np

from waveop import causal, sprinkle_11
from crown import compat, triangles

N = 2000


def links_of(R):
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    return R & (Iv == 0)


def closure(R):
    C = R.copy()
    for _ in range(60):
        nx = C | ((C.astype(np.float32) @ C.astype(np.float32)) > 0.5)
        if (nx == C).all():
            break
        C = nx
    return C


def truncate(L, P, k):
    """оставить каждому элементу k ближайших связей В ПРОШЛОЕ."""
    n = L.shape[0]
    u, v = P[:, 0], P[:, 1]
    out = np.zeros_like(L)
    for j in range(n):
        ys = np.nonzero(L[:, j])[0]
        if len(ys) == 0:
            continue
        if len(ys) <= k:
            out[ys, j] = True
            continue
        tau = (u[j] - u[ys]) * (v[j] - v[ys])
        sel = ys[np.argsort(tau)[:k]]
        out[sel, j] = True
    return out


def stats(R, rs, lab):
    n = R.shape[0]
    r = float(R.sum()) / (n * (n - 1) / 2)
    L = links_of(R)
    lk = float(L.sum()) / n
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    ii, jj = np.nonzero(R & (Iv >= 12))
    inn = float('nan')
    if len(ii):
        sel = rs.choice(len(ii), min(100, len(ii)), replace=False)
        vv = []
        for t in sel:
            i, j = ii[t], jj[t]
            m = np.nonzero(R[i] & R[:, j])[0]
            if len(m) >= 3:
                vv.append(float(R[np.ix_(m, m)].sum())
                          / (len(m) * (len(m) - 1) / 2))
        inn = float(np.mean(vv)) if vv else float('nan')
    G = compat(R, rs, m=1200)
    tri = triangles(G)[0] if G is not None else float('nan')
    print(f'{lab:>18}{r:>10.4f}{inn:>12.4f}{lk:>12.2f}{tri:>12.1f}')
    return r, inn, lk, tri


def main():
    print(__doc__)
    rs = np.random.default_rng(5150)
    P = sprinkle_11(N, rs)
    R0 = causal(P)
    L0 = links_of(R0)
    print(f'засев 1+1, N={N}')
    print(f"{'что':>18}{'доля связ.':>10}{'внутр.доля':>12}"
          f"{'связей/эл.':>12}{'корон S_3':>12}")
    stats(R0, rs, 'исходный')
    for k in (10 ** 9, 12, 8, 5, 3, 2, 1):
        T = closure(truncate(L0, P, k))
        stats(T, rs, f'обрезано до {k}' if k < 100 else 'контроль k=inf')


if __name__ == '__main__':
    main()
