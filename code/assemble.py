#!/usr/bin/env python3
"""СБОРКА: квантовый рост на фазах из порядка, без подогнанных чисел.

ЧТО СКЛАДЫВАЕТСЯ.

  1. Из архива (`ORDER-final.md:1994`): новое событие выбирает предков с
     вероятностью |A|^2, где A — сумма амплитуд. Период ровно 2*pi, при
     pi полное гашение. Но мир выходил вырожденным (размерность 1.4-1.6),
     и вилка снималась ЧЕРЕДОВАНИЕМ с выравнивающим шагом, у которого
     доля срезания 0.05 ПОДОБРАНА РУКАМИ.

  2. Из архива (`compute-doc.md:155`): амплитуду не нужно приписывать,
     она строится из самих фаз: A = exp(2*pi*i*sum_m phi_m), потому что
     k чисел события периодичны. И нормировать на каждом шаге НЕЛЬЗЯ —
     нормировка стирает относительные фазы (линейность нарушалась на
     3.27, интерференция обращалась в нуль).

  3. Из сегодняшнего (`statgrow-doc.md`): шкалы надо брать порядковыми
     отношениями счётов. Здесь фазы — доля прошлого, общая с опорой:

         phi_m(v) = |past(v) & past(r_m)| / |past(v)|

     безразмерна, лежит в [0,1) сама, параметра не требует. И считается
     ОДИН РАЗ при рождении: новое событие максимально, значит прошлые
     уже рождённых не меняются.

  4. Из сегодняшнего (`sorkin-doc.md`): интерференция душит плотный
     сектор в 17..100 раз БЕЗ параметра. А выравнивающий шаг архива
     делал именно это. Причина арифметическая: у множества из |D|
     предков со случайными фазами |A|^2 ~ |D|, а не |D|^2, то есть
     интерференция сама штрафует плотное прицепление.

ГИПОТЕЗА ОДНИМ ПРОХОДОМ: если так, подогнанный выравнивающий шаг не
нужен, и пять различителей пройдутся без него.

СЕМЕЙСТВА: квант (|A|^2), равновер. (контроль), богатые (по числу
потомков — архивное «богатые богатеют»).

  python3 assemble.py
  python3 assemble.py --ns 400 800 --k 4 --seeds 3
"""
import argparse
import math
import time

import numpy as np

from core import frac_pairs, mm_dim, height, spectral, links_of
from core import transitivity_violations


def grow(N, k, rule, ncand, rs, seed_n=12):
    """Рост по одному событию. Pm[i,j] = True, если j раньше i."""
    Pm = np.zeros((N, N), bool)
    amp = np.ones(N, complex)
    refs = None
    refpast = None
    for n in range(1, N):
        # кандидаты: вниз-замкнутые подмножества (прошлые случайных событий)
        cands = [np.empty(0, np.int64)]
        for _ in range(ncand):
            m = int(rs.integers(1, 4))
            mask = np.zeros(n, bool)
            for v in rs.integers(0, n, m):
                v = int(v)
                mask[v] = True
                mask |= Pm[v, :n]
            cands.append(np.flatnonzero(mask))
        # убрать повторы
        seen, uniq = set(), []
        for c in cands:
            key = (len(c), int(c.sum()) if len(c) else -1)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        cands = uniq

        if n < seed_n or refs is None or rule == 'равновер.':
            pick = int(rs.integers(0, len(cands)))
        elif rule == 'калибр':
            # КАЛИБРОВОЧНАЯ ФАЗА как амплитуда шага.
            #
            # chirphase.py: показатель |A|^2 ~ |D|^alpha вышел
            #   чётность из истинных шкал   0.961  (работает)
            #   чётность из порядковых      2.000  (когерентна)
            #   КАЛИБРОВОЧНАЯ ФАЗА          1.312  (работает, и она
            #                                      чисто порядковая)
            #   гладкая из порядка          2.000
            #   случайные                   1.060
            #
            # Причина: источник интерференции — ХОЛОНОМИЯ. Фаза, которая
            # телескопируется, по множеству сводится к границе и
            # интерферировать не может. У калибровочной холономия
            # 22.24 +- 2.54 (8.7 сигмы), а «разность прошлого» на том же
            # коде даёт ровно нуль (ORDER-final.md:3508).
            #
            # Для нового МАКСИМАЛЬНОГО события n с прошлым D:
            #   phi(v, n) = |общие предки| - |общие потомки|
            #             = |D & past(v)| - 0
            # то есть число элементов D под v. Считается по кандидату.
            w = []
            for c in cands:
                if len(c) == 0:
                    w.append(1.0); continue
                sub = Pm[np.ix_(c, c)]
                g = sub.sum(1)                 # |D & past(v)| для v из D
                w.append(abs(np.exp(1j * g).sum()) ** 2)
            w = np.array(w)
            if w.sum() <= 0:
                pick = int(rs.integers(0, len(cands)))
            else:
                pick = int(rs.choice(len(cands), p=w / w.sum()))
        elif rule == 'калибр/N':
            # ТО ЖЕ, но вес делится на число предков.
            #
            # ПОЧЕМУ. При показателе alpha ~ 1.3 величина |A|^2 РАСТЁТ с
            # числом предков — больше слагаемых, больше сумма, — значит
            # вес кандидата пропорционален его размеру и правило
            # поощряет плотное прицепление просто счётом членов. Я
            # выбросил из амплитуды множитель затухания hop, решив, что
            # он уйдёт в нормировку; он уходит только если у всех
            # кандидатов одинаковый размер.
            #
            # Архив в пропагаторе делил на ЧИСЛО ПУТЕЙ, чтобы увидеть
            # затухание (0.250, 0.218, 0.049, 0.007 —
            # propagator-doc.md). Здесь то же.
            #
            # ЧЕСТНО: это ВЫБОР НОРМИРОВКИ, а не вывод из аксиом.
            # Квантовая версия аксиом Райдаута-Соркина (дискретная общая
            # ковариантность + причинность Белла) в области открыта.
            w = []
            for c in cands:
                if len(c) == 0:
                    w.append(1.0); continue
                sub = Pm[np.ix_(c, c)]
                g = sub.sum(1)
                w.append(abs(np.exp(1j * g).sum()) ** 2 / len(c))
            w = np.array(w)
            if w.sum() <= 0:
                pick = int(rs.integers(0, len(cands)))
            else:
                pick = int(rs.choice(len(cands), p=w / w.sum()))
        elif rule == 'квант':
            w = np.array([abs(amp[c].sum()) ** 2 if len(c) else 0.0
                          for c in cands])
            # пустой выбор оставляем возможным с весом одного события
            w[0] = 1.0
            if w.sum() <= 0:
                pick = int(rs.integers(0, len(cands)))
            else:
                pick = int(rs.choice(len(cands), p=w / w.sum()))
        else:                                   # богатые богатеют
            w = np.array([float(len(c)) + 1.0 for c in cands])
            pick = int(rs.choice(len(cands), p=w / w.sum()))

        D = cands[pick]
        if len(D):
            Pm[n, D] = True
            Pm[n, :n] |= Pm[D, :n].any(0)
        # опоры: первые k событий после затравки
        if refs is None and n >= k:
            refs = np.arange(k)
            refpast = [Pm[int(r), :].copy() for r in refs]
            for r in refs:
                refpast[int(r)] = Pm[int(r)].copy()
        # фаза нового события — один раз, из отношений счётов
        if refs is not None:
            p_n = Pm[n]
            sz = int(p_n.sum())
            if sz == 0:
                amp[n] = 1.0
            else:
                s = 0.0
                for m_ in range(k):
                    s += float((p_n & refpast[m_]).sum()) / sz
                amp[n] = np.exp(2j * np.pi * s)
    return Pm


def local_dim(Pm, rs, nsub=8, frac=0.45):
    """Размерность подинтервалов: однородность по масштабу, 5-й различитель."""
    N = len(Pm)
    out = []
    for _ in range(nsub * 6):
        if len(out) >= nsub:
            break
        a, b = sorted(rs.choice(N, 2, replace=False))
        if not Pm[b, a]:
            continue
        mid = np.flatnonzero(Pm[:, a] & Pm[b, :])
        if len(mid) < max(30, int(frac * 0.1 * N)):
            continue
        sub = Pm[np.ix_(mid, mid)]
        out.append(mm_dim(frac_pairs(sub)))
    return float(np.nanmean(out)) if out else float('nan')


def main():
    ap = argparse.ArgumentParser(description='Сборка: квантовый рост')
    ap.add_argument('--ns', type=int, nargs='+', default=[400, 800])
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--cands', type=int, default=18)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--rules', nargs='+',
                    default=['калибр', 'квант', 'равновер.'])
    a = ap.parse_args()

    t0 = time.time()
    print(f'фаз k={a.k}, кандидатов ~{a.cands}, засевов {a.seeds}')
    print('фаза = sum_m |past(v) & past(r_m)| / |past(v)|, '
          'амплитуда exp(2*pi*i*фаза)')
    print('ни одного подогнанного числа; выравнивающего шага НЕТ')
    print()
    print('КРИТЕРИЙ ГОДНОСТИ (PICTURE.md §5):')
    print('  наруш. 0 | доля 0.098-0.103 | разм. 4.00 | спектр > 0 и растёт')
    print('  связ/соб ~ N^0.75 | Жаккар 0.169 | локальная = глобальной')
    print()
    print(f"{'правило':>11}{'N':>6}{'доля пар':>10}{'разм.':>7}{'высота':>8}"
          f"{'спектр':>10}{'связ/соб':>9}{'Жаккар':>8}{'лок.разм':>9}"
          f"{'наруш':>7}")

    tab = {}
    for rule in a.rules:
        for N in a.ns:
            rows = []
            for t in range(a.seeds):
                rs = np.random.default_rng(5101000 + 977 * t)
                Pm = grow(N, a.k, rule, a.cands, rs)
                f = frac_pairs(Pm)
                L = links_of(Pm)
                A = Pm.astype(np.float32)
                CP = A @ A.T
                sz = A.sum(1)
                iu = np.triu_indices(N, 1)
                jac = float(np.mean(
                    CP[iu] / np.maximum(sz[:, None] + sz[None, :] - CP, 1)[iu]))
                rows.append((f, mm_dim(f), height(Pm)[0], spectral(Pm),
                             L.sum() / N, jac,
                             local_dim(Pm, rs),
                             transitivity_violations(Pm)))
                del Pm
            M = np.array([r[:7] for r in rows], float)
            m = np.nanmean(M, 0)
            tab[(rule, N)] = m
            print(f'{rule:>11}{N:>6}{m[0]:>10.4f}{m[1]:>7.2f}{m[2]:>8.1f}'
                  f'{m[3]:>+10.4f}{m[4]:>9.2f}{m[5]:>8.4f}{m[6]:>9.2f}'
                  f'{int(rows[0][7]):>7}', flush=True)
        print()

    print('УСТОЙЧИВОСТЬ ПО N и показатели')
    for rule in a.rules:
        if len(a.ns) < 2:
            continue
        m1, m2 = tab[(rule, a.ns[0])], tab[(rule, a.ns[-1])]
        lr = math.log(a.ns[-1] / a.ns[0])
        ph = math.log(max(m2[2], 1e-9) / max(m1[2], 1e-9)) / lr
        pl = math.log(max(m2[4], 1e-9) / max(m1[4], 1e-9)) / lr
        print(f'  {rule:>11}: доля {m1[0]:.4f}->{m2[0]:.4f}, '
              f'разм. {m1[1]:.2f}->{m2[1]:.2f}, '
              f'спектр {m1[3]:+.4f}->{m2[3]:+.4f}, '
              f'показ. высоты {ph:.3f} (эталон 0.279), '
              f'показ. связей {pl:.3f} (эталон 0.75)')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
