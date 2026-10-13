#!/usr/bin/env python3
"""СКЛЕЙКА: правило как ЯЗЫК, а не как мера.

Догадка: полоса N ~ 100…350, в которой ломаются ВСЕ наши маршруты, есть
не место поломки, а размер БЛОКА, на котором правило впервые выразимо.
Четыре независимых расчёта дают её: запрещённые подпорядки начинаются
к n ~ 108 (счёт Уоррена), фазовый переход 74…178, окно меры роста
180…220, смена знака дефицита 320. А целое собирается склейкой двух
блоков «раньше-позже» с условием согласования на интерфейсе.

Это не мера на конфигурациях, а ЯЗЫК: что разрешено приклеивать к чему.
Ни в одном закрытом классе такого нет.

Лаборатория — d = 2, где реализуемость есть РОВНО «порядковая
размерность <= 2», и это проверяется точно. Склейка: C = A + B с
внутренними порядками блоков плюс интерфейс I ⊆ A×B, замкнутый
транзитивностью.

ЧЕТЫРЕ ТОЧНЫХ ВОПРОСА:
  1. Ограничена ли склейка вообще? (Есть ли законные блоки и законный
     интерфейс, дающие НЕзаконное целое.) Предсказание: да, корона.
  2. Насколько часто склейка незаконна — перебором всех блоков и всех
     замкнутых интерфейсов малого размера.
  3. Конечен ли список запретов? Считаю минимальные порядки
     размерности 3 при n = 5, 6, 7: если число растёт, список
     бесконечен, и правило в одном порядке локальным быть не может.
  4. СТАНОВИТСЯ ЛИ ПРАВИЛО ЛОКАЛЬНЫМ СО СКРЫТЫМ СОСТОЯНИЕМ. Если
     каждый блок несёт свои две координаты (пару линейных порядков,
     пересечение которых даёт блок), то законность интерфейса обязана
     определяться ТОЛЬКО сравнением координат на интерфейсе. Проверяю
     равносильность: «склейка 2-мерна» <=> «существуют представления
     блоков, согласующиеся на интерфейсе». Если равносильность точна,
     скрытое состояние есть координаты, то есть геометрия.

КОНТРОЛЬ dim2: алгоритм точный, не эвристика. P = L1 ∩ L2 означает,
что L2 ОБЯЗАН обратить каждую несравнимую пару относительно L1, то
есть L2 определён однозначно по L1; значит достаточно перебрать
линейные расширения L1 и проверить транзитивность вынужденного L2.
Сверяется с известными фактами: антицепь размерности 2 при k>=2,
цепь — 1, корона 3+3 — ровно 3.
"""
import itertools

import numpy as np


def closure(R):
    n = len(R)
    R = R.copy()
    for k in range(n):
        R |= R[:, k][:, None] & R[k]
    return R


def lin_exts(R):
    n = len(R)
    out = []
    for p in itertools.permutations(range(n)):
        pos = np.empty(n, int)
        pos[list(p)] = np.arange(n)
        ok = True
        for i in range(n):
            for j in range(n):
                if R[i, j] and pos[i] > pos[j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(pos)
    return out


def dim_le_2(R, want_reps=False):
    """СЛОМАНА — НЕ ИСПОЛЬЗОВАТЬ. Берите compose2.dim_le_2.

    Ориентация L2 перевернулась при «оптимизации»: функция объявляет
    двумерной корону 3+3, у которой размерность ровно 3. Поймано
    контролем короны дважды — 10 и 11 октября (во второй раз я сам
    импортировал её сюда и получил «доля dim<=2 равна 1.000»).
    Оставлена только чтобы старые ссылки не падали молча.
    """
    raise RuntimeError(
        'compose.dim_le_2 сломана (корона 3+3 проходит как двумерная); '
        'используйте compose2.dim_le_2')


def _dim_le_2_broken(R, want_reps=False):
    """Точно: существуют ли L1, L2 с P = L1 ∩ L2."""
    n = len(R)
    reps = []
    for pos1 in lin_exts(R):
        # вынужденный второй порядок
        less2 = np.zeros((n, n), bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if R[i, j]:
                    less2[i, j] = True
                elif not R[j, i]:
                    less2[i, j] = pos1[j] < pos1[i]
        if (less2 & less2.T).any():
            continue
        if (closure(less2.copy()) != less2).any():
            continue
        if (np.count_nonzero(less2) != n * (n - 1) // 2):
            continue
        pos2 = np.empty(n, int)
        pos2[np.argsort(less2.sum(1))[::-1]] = np.arange(n)
        if want_reps:
            reps.append((pos1.copy(), pos2))
        else:
            return True, None
    return (len(reps) > 0), (reps if want_reps else None)


def glue(RA, RB, I):
    a, b = len(RA), len(RB)
    R = np.zeros((a + b, a + b), bool)
    R[:a, :a] = RA
    R[a:, a:] = RB
    R[:a, a:] = I
    return closure(R)


def closed_interfaces(RA, RB):
    """Интерфейсы, замкнутые транзитивностью внутри блоков."""
    a, b = len(RA), len(RB)
    for bits in range(1 << (a * b)):
        I = np.array([(bits >> k) & 1 for k in range(a * b)],
                     bool).reshape(a, b)
        if (RA @ I & ~I).any() or (I @ RB & ~I).any():
            continue
        yield I


def antichain(k):
    return np.zeros((k, k), bool)


def crown():
    R = np.zeros((6, 6), bool)
    for i in range(3):
        for j in range(3):
            if i != j:
                R[i, 3 + j] = True
    return R


def main():
    print(__doc__)
    print('КОНТРОЛЬ dim2 на известных ответах')
    ch = np.zeros((4, 4), bool)
    for i in range(4):
        for j in range(i + 1, 4):
            ch[i, j] = True
    for nm, R, exp in (('цепь из 4', ch, True),
                       ('антицепь из 4', antichain(4), True),
                       ('корона 3+3', crown(), False)):
        got, _ = dim_le_2(R)
        print(f'  {nm:>14}: размерность <= 2? {got}  (ожидалось {exp})'
              f'{"" if got == exp else "   <-- ОШИБКА"}')

    print('\nВОПРОС 1: ограничена ли склейка')
    A, B = antichain(3), antichain(3)
    I = np.array([[i != j for j in range(3)] for i in range(3)])
    C = glue(A, B, I)
    print(f'  блок A (антицепь 3): dim<=2 {dim_le_2(A)[0]}')
    print(f'  блок B (антицепь 3): dim<=2 {dim_le_2(B)[0]}')
    print(f'  интерфейс замкнут: '
          f'{not ((A @ I & ~I).any() or (I @ B & ~I).any())}')
    print(f'  СКЛЕЙКА: dim<=2 {dim_le_2(C)[0]}  '
          f'=> склейка {"ОГРАНИЧЕНА" if not dim_le_2(C)[0] else "свободна"}')

    print('\nВОПРОС 2: как часто склейка законных блоков незаконна')
    print(f"{'|A|':>4}{'|B|':>4}{'блоков A':>10}{'интерфейсов':>13}"
          f"{'незаконных склеек':>19}{'доля':>8}")
    from realiz import all_posets
    for a, b in ((2, 3), (3, 3), (2, 4)):
        PA = [R for R in all_posets(a).values() if dim_le_2(R)[0]]
        PB = [R for R in all_posets(b).values() if dim_le_2(R)[0]]
        tot = bad = 0
        for RA in PA:
            for RB in PB:
                for I in closed_interfaces(RA, RB):
                    tot += 1
                    if not dim_le_2(glue(RA, RB, I))[0]:
                        bad += 1
        print(f'{a:>4}{b:>4}{len(PA):>10}{tot:>13}{bad:>19}'
              f'{bad/max(tot,1):>8.3f}')

    growth = []
    print('\nВОПРОС 3: конечен ли список запретов '
          '(минимальные порядки размерности 3)')
    for n in (5, 6, 7):
        ps = list(all_posets(n).values())
        bad3 = [R for R in ps if not dim_le_2(R)[0]]
        mini = 0
        for R in bad3:
            ok = True
            for drop in range(n):
                keep = [i for i in range(n) if i != drop]
                if not dim_le_2(R[np.ix_(keep, keep)])[0]:
                    ok = False
                    break
            if ok:
                mini += 1
        print(f'  n={n}: порядков {len(ps)}, размерности >=3 {len(bad3)}, '
              f'из них МИНИМАЛЬНЫХ {mini}', flush=True)
        growth.append((n, mini))
    if len(growth) >= 2:
        seq = ', '.join(f'n={n}: {m}' for n, m in growth)
        print(f'  минимальных по n: {seq}')
        print('  => список запретов '
              + ('РАСТЁТ, значит конечным быть не может, и правило в '
                 'одном порядке локальным не бывает'
                 if growth[-1][1] > growth[0][1] else
                 'не растёт на этом размахе — вывод преждевременный'))
    q4(all_posets)


def q4(all_posets):
    """ШАГ РОСТА: сколько прошлых-подмножеств сохраняют законность.

    Добавление одного элемента b с прошлым D (нижнее множество A) —
    это ровно шаг Райдаута–Соркина. Вопрос: какая доля шагов сохраняет
    двумерность, и стабильна ли она. Если доля держится — язык
    «растим», и у локального правила есть место. Если обваливается —
    законность становится глобальным сговором, и роста не выйдет.
    """
    print('\nВОПРОС 4: шаг роста — доля законных прошлых')
    print(f"{'|A|':>5}{'блоков':>8}{'шагов всего':>13}"
          f"{'законных':>10}{'доля':>8}{'мин. доля по блоку':>20}")
    for a in (2, 3, 4, 5):
        PA = [R for R in all_posets(a).values() if dim_le_2(R)[0]]
        tot = good = 0
        worst = 1.0
        for RA in PA:
            t = g = 0
            # нижние множества: D замкнуто вниз
            for bits in range(1 << a):
                D = np.array([(bits >> i) & 1 for i in range(a)], bool)
                if (RA & D[None, :] & ~D[:, None]).any():
                    continue          # есть x<y, y в D, x не в D
                R = np.zeros((a + 1, a + 1), bool)
                R[:a, :a] = RA
                R[:a, a] = D
                t += 1
                if dim_le_2(closure(R))[0]:
                    g += 1
            tot += t
            good += g
            worst = min(worst, g / max(t, 1))
        print(f'{a:>5}{len(PA):>8}{tot:>13}{good:>10}'
              f'{good/max(tot,1):>8.3f}{worst:>20.3f}', flush=True)


if __name__ == '__main__':
    main()
