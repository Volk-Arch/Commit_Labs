#!/usr/bin/env python3
"""КАРТА ОСТАТКОВ: где данные расходятся с выведенным предсказанием.

Переподгонка (`refit.py`) дала отношение замеренного к истине: lin 0.87,
quad 0.93, exp 0.54, root 0.50 — все НИЖЕ единицы, то есть системно.
Полоса середин (`|mid - eta_0| <= 1.6`) объясняет лишь +-12%: у `root`
кривизна идёт как `1/a^4`, у `lin` как `1/a^2`, и на этой полосе `a`
меняется на 5%. Для двукратного дефекта мало.

Алгебра проверена вручную и согласована, точный интеграл её подтвердил
(`vcal2.py`, 0.2 сигмы на `exp`). Значит расхождение не внутри теории, а
между формулой и ОТОБРАННЫМИ данными: набор берёт интервалы по квотам,
применяет защиту применимости `tau² R_uu <= 0.8` и требует `C_1` в
полосе. Любой отбор, скоррелированный с `X_2`, смещает среднее.

ЗДЕСЬ для каждой строки считается предсказание из ИСТИННЫХ компонент
и её собственных `v²`, `C_1`:

    pred = [ A(v²) R_etaeta + B(v²) R_xx ] * KFAC * sqrt(C_1)

и отношение `замер/предсказание` печатается по ячейкам `v² x C_1`.

ЧТЕНИЕ. Если отношение РОВНОЕ по ячейкам — дело в общем множителе
(перевод `tau² <-> C_1`, или множители `a`). Если растёт с `v²` или с
`C_1` — дело в форме профиля или в отборе.

  python3 resid.py
"""
import math

import numpy as np

from pencil import KFAC, truth, IC, IX2, IX3, IV, IM
from refit import load_all, VT, A2, B2, A3, B3

VB = (0.0, 0.2, 0.4, 0.6, 0.85001)
CB = (200.0, 380.0, 560.0, 900.0001)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    arms = load_all()
    for fam in ('lin', 'quad', 'exp', 'root'):
        A, sd, par = arms[fam]
        tr = truth(fam, par)
        t2 = KFAC * np.sqrt(A[:, IC])
        for col, Aт, Bт, nm in ((IX2, A2, B2, 'X_2'), (IX3, A3, B3, 'X_3')):
            pred = (np.interp(A[:, IV], VT, Aт) * tr[0]
                    + np.interp(A[:, IV], VT, Bт) * tr[1]) * t2
            print(f'{fam} {nm}: предсказано в среднем {pred.mean():+.5f}, '
                  f'замерено {A[:, col].mean():+.5f}, '
                  f'отношение {A[:, col].mean()/pred.mean():.3f}')
            if nm == 'X_2':
                print(f"{'':>8}", end='')
                for j in range(len(CB) - 1):
                    print(f'{f"C1 {CB[j]:.0f}-{CB[j+1]:.0f}":>16}', end='')
                print()
                for i in range(len(VB) - 1):
                    print(f'{f"v² {VB[i]:.1f}-{VB[i+1]:.1f}":>8}', end='')
                    for j in range(len(CB) - 1):
                        m = ((A[:, IV] >= VB[i]) & (A[:, IV] < VB[i + 1])
                             & (A[:, IC] >= CB[j]) & (A[:, IC] < CB[j + 1]))
                        if m.sum() < 80:
                            print(f'{"—":>16}', end=''); continue
                        us = np.unique(sd[m])
                        jk = np.array([A[m & (sd != u), col].mean()
                                       for u in us])
                        se = math.sqrt((len(us) - 1) / len(us)
                                       * ((jk - jk.mean()) ** 2).sum())
                        r = A[m, col].mean() / pred[m].mean()
                        er = se / abs(pred[m].mean())
                        print(f'{f"{r:.2f}+-{er:.2f}":>16}', end='')
                    print(flush=True)
        print()


if __name__ == '__main__':
    main()
