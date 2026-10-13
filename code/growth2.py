#!/usr/bin/env python3
"""Масштабно-инвариантные правила роста против нынешнего критерия.

ПОВОД. growth.py провалился с точным диагнозом: прошлое нового события
держалось около шести событий при любом N, тогда как нужна постоянная
ДОЛЯ мира. Правило давало постоянную валентность — дерево, не
многообразие. И результат зависел от размера моего пула (бистабильность).

АРХИВ. Там уже есть масштабно-инвариантное правило — «богатые богатеют»:
новое событие присоединяется к тем, у кого прошлое больше. При N=1000
оно дало долю 0.1043 +- 0.0013 и размерность 3.95 +- 0.01, и S_n не рос.
Но проверялось оно по СТАРОМУ критерию, калиброванному на покомпонентном
ящике, где хорошим спектром считался -0.008. Нынешний эталон — конусный
Минковский, и он требует спектр ПОЛОЖИТЕЛЬНЫЙ (+0.0162 при N=800).
Плюс тогда не было двух новых пунктов: доля прошлого и валентность.

ГЕНЕРАТОРЫ (все дают замкнутое вниз прошлое, значит транзитивность
выполняется по построению).

  psub    случайное подмножество доли p, затем замыкание вниз
          (контроль: это генератор из growth.py, он и провалился)
  rich    то же, но вероятность попасть в затравку пропорциональна
          размеру прошлого — «богатые богатеют», масштабно-инвариантно
  union   объединение прошлых m случайных событий
  recent  объединение прошлых m недавно добавленных

КРИТЕРИЙ (action-weight-doc.md + step-phase-doc.md), эталон — конус:

  доля пар                 около 0.10
  размерность              4.00, УСТОЙЧИВАЯ по размеру мира
  высота                   около 9 при N=800 (растёт как N^0.25)
  спектр                   ПОЛОЖИТЕЛЬНЫЙ
  нарушений транзитивности ровно 0
  <|прошлое|>/N            постоянно (около 0.05), не постоянное число
  связей на событие        растёт как N^0.75
  и ничего не зависит от параметров самой выборки

  python3 growth2.py
  python3 growth2.py --ns 400 800 1600 3200
"""
import argparse
import math
import time

import numpy as np

from core import (mm_dim, height, spectral, transitivity_violations,
                  frac_pairs, links_of)
from weight import w_of


def seeds_psub(k, p, past, rs, K):
    return rs.random((k, K)) < p


def seeds_rich(k, p, past, rs, K):
    w = past.astype(np.float64) + 1.0
    q = p * k * w / w.sum()                 # ожидаемый размер затравки p*k
    return rs.random((k, K)) < np.minimum(q, 1.0)[:, None]


def seeds_desc(k, p, past, rs, K):
    """Архивное ПРАВИЛО: отбор по числу ПОТОМКОВ (m = p*k без возврата)."""
    w = _DESC[:k].astype(np.float64) + 1.0
    w /= w.sum()
    m = min(max(1, int(p * k)), k)
    S = np.zeros((k, K), bool)
    for j in range(K):
        S[rs.choice(k, m, replace=False, p=w), j] = True
    return S


def seeds_richsel(k, p, past, rs, K):
    """То же, но по размеру ПРОШЛОГО — как написано в названии архива."""
    w = past.astype(np.float64) + 1.0
    w /= w.sum()
    m = min(max(1, int(p * k)), k)
    S = np.zeros((k, K), bool)
    for j in range(K):
        S[rs.choice(k, m, replace=False, p=w), j] = True
    return S


def seeds_alpha(k, p, past, rs, K):
    """Ослабленное предпочтение: вес ~ (1+потомки)^ALPHA.

    ALPHA=0 — равномерно (перколяция, без обратной связи),
    ALPHA=1 — правило desc из архива (полная обратная связь).
    """
    w = (_DESC[:k].astype(np.float64) + 1.0) ** _ALPHA
    w /= w.sum()
    m = min(max(1, int(p * k)), k)
    S = np.zeros((k, K), bool)
    for j in range(K):
        S[rs.choice(k, m, replace=False, p=w), j] = True
    return S


def seeds_union(k, p, past, rs, K):
    m = max(1, int(round(p * k)))
    m = min(m, 6)
    S = np.zeros((k, K), bool)
    for j in range(K):
        S[rs.integers(0, k, m), j] = True
    return S


def seeds_recent(k, p, past, rs, K):
    m = max(1, min(6, int(round(p * k))))
    lo = max(0, k - max(4, k // 4))
    S = np.zeros((k, K), bool)
    for j in range(K):
        S[rs.integers(lo, k, m), j] = True
    return S


GENS = dict(psub=seeds_psub, rich=seeds_rich, union=seeds_union,
            recent=seeds_recent, desc=seeds_desc,
            richsel=seeds_richsel, alpha=seeds_alpha)
_ALPHA = 1.0
_DESC = np.zeros(1, np.int64)


def grow(N, gen, p, beta, eps, K, rs):
    global _DESC
    R = np.zeros((N, N), bool)
    _DESC = np.zeros(N, np.int64)
    wtab = w_of(np.arange(N + 2), eps)
    past = np.zeros(N, np.int64)
    sizes = []
    for k in range(1, N):
        A = R[:k, :k].astype(np.float32)
        S = GENS[gen](k, p, past[:k], rs, K)
        D = S | ((A.T @ S.astype(np.float32)) > 0)
        if beta == 0.0:
            j = int(rs.integers(K))
        else:
            Df = D.astype(np.float32)
            n = np.rint(A.T @ Df).astype(np.int32)
            dS = eps + (wtab[np.clip(n, 0, N + 1)] * D).sum(0)
            lw = -beta * dS
            lw -= lw.max()
            pr = np.exp(lw)
            j = int(rs.choice(K, p=pr / pr.sum()))
        d = D[:, j]
        R[k, :k] = d
        past[k] = int(d.sum())
        _DESC[:k] += d            # у всех в прошлом стало на потомка больше
        sizes.append(past[k] / k)
    return R, float(np.mean(sizes[len(sizes) // 2:]))


def main():
    ap = argparse.ArgumentParser(description='Масштабно-инвариантный рост')
    ap.add_argument('--ns', type=int, nargs='+', default=[400, 800, 1600])
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--k', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--p', type=float, default=0.05)
    ap.add_argument('--betas', type=float, nargs='+', default=[0.0, 100.0])
    ap.add_argument('--gens', type=str, nargs='+',
                    default=['psub', 'rich', 'union', 'recent'])
    a = ap.parse_args()

    t0 = time.time()
    print(f'eps={a.eps}, кандидатов {a.k}, p={a.p}, засевов {a.seeds}')
    print('ЭТАЛОН (конус Минковского): доля 0.10, размерн. 4.00, '
          'спектр > 0,')
    print('высота ~9 при N=800, <|прошлое|>/N ~0.05, наруш. 0')
    print()
    hdr = (f"{'':>18}{'доля пар':>17}{'размерность':>15}{'высота':>7}"
           f"{'спектр':>10}{'наруш':>7}{'|прош|/N':>10}{'связ/соб':>10}")
    for gen in a.gens:
        for beta in a.betas:
            print(f'  {gen}, beta={beta:g}')
            print(hdr)
            lk = []
            for N in a.ns:
                acc = []
                for t in range(a.seeds):
                    rs = np.random.default_rng(5500 + 17 * t)
                    R, ps = grow(N, gen, a.p, beta, a.eps, a.k, rs)
                    f = frac_pairs(R)
                    acc.append((f, mm_dim(f), height(R)[0], spectral(R),
                                transitivity_violations(R), ps,
                                links_of(R).sum() / N))
                    del R
                A = np.array(acc, float)
                m, e = A.mean(0), A.std(0) / math.sqrt(a.seeds)
                lk.append(m[6])
                print(f'{f"N={N}":>18}{m[0]:>10.4f} ±{e[0]:<6.4f}'
                      f'{m[1]:>9.3f} ±{e[1]:<5.3f}{m[2]:>7.1f}'
                      f'{m[3]:>+10.4f}{int(m[4]):>7}{m[5]:>10.4f}'
                      f'{m[6]:>10.1f}', flush=True)
            if len(a.ns) > 1:
                pe = np.polyfit(np.log(a.ns), np.log(np.maximum(lk, 1e-9)),
                                1)[0]
                print(f'{"показатель связей по N":>28}: {pe:.3f}'
                      f'   (цель 0.75)')
            print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
