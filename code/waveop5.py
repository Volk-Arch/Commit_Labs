#!/usr/bin/env python3
"""ОПЕРАТОР 1+1, исправленный по диагностике.

waveid.py замерил на один элемент <sum_m w_m n_m> = 0.480, 0.456,
0.543, 0.598 при весах (1, -2, 1) — то есть около 1/2, а не 1. Значит
скобка [-f(x) + sum w n f] равна (-1 + 1/2) f = -f/2, не ноль, и
оператор расходится как rho. Это ровно то, что замерено
(-148, -346, -728 ~ -0.4n).

Исправление, найденное ИЗМЕРЕНИЕМ, а не памятью: собственный
коэффициент есть -1/2.

    (B f)(x) = (1/l^2) [ -1/2 f(x) + sum_{y<x} (1,-2,1) f(y) ]

КОНТРОЛЬ: на гладких функциях среднее обязано идти к точным 2, -2, 4
(при сигнатуре (+,-): кв(t^2)=2, кв(x^2)=-2, кв(t^2-x^2)=4), и ошибки
печатаются — прошлый раз я их не напечатал в 4D и чуть не объявил
оператор неверным по шуму.
"""
import math

import numpy as np

from waveop import causal, sprinkle_11

W11 = (1.0, -2.0, 1.0)


def Bmat(Rc, rho, cself):
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    B = np.zeros((n, n))
    for m, w in enumerate(W11):
        B += w * (Rc.T & (Iv.T == m))
    np.fill_diagonal(B, cself)
    return B * rho                      # 1/l^2 = rho в 1+1


def main():
    print(__doc__)
    rs = np.random.default_rng(27182)
    for cself in (-1.0, -0.5):
        print(f'\n=== собственный коэффициент {cself} ===')
        print(f"{'N':>6}{'засевов':>9}{'точек':>8}"
              f"{'<B t²>':>10}{'±':>8}{'<B x²>':>10}{'±':>8}"
              f"{'<B τ²>':>10}{'±':>8}")
        for n, nspr in ((400, 90), (900, 45), (1800, 20)):
            acc = [[], [], []]
            for _ in range(nspr):
                P = sprinkle_11(n, rs)
                rho = n / 0.5
                B = Bmat(causal(P), rho, cself)
                t, x = P[:, 0], P[:, 1]
                core = (t > 0.3) & (t < 0.7) & (np.abs(x) < 0.15)
                if core.sum() < 2:
                    continue
                for i, f in enumerate((t * t, x * x, t * t - x * x)):
                    acc[i].extend(((B @ f)[core]).tolist())
            out = []
            for a in acc:
                v = np.array(a)
                out.append((float(v.mean()),
                            float(v.std() / math.sqrt(len(v)))))
            print(f'{n:>6}{nspr:>9}{len(acc[0]):>8}'
                  + ''.join(f'{m:>10.2f}{e:>8.2f}' for m, e in out),
                  flush=True)
        print('   точные ответы: 2, -2, 4')


if __name__ == '__main__':
    main()
