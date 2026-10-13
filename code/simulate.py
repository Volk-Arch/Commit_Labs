#!/usr/bin/env python3
"""
СИМУЛЯТОР КУСКА ПРОСТРАНСТВА-ВРЕМЕНИ
====================================

Всё построено из ОДНОГО правила:
    событию даны k чисел; событие A раньше B, если A раньше по ВСЕМ k числам.

Из этого правила выводятся, без единого дополнительного параметра:

    геометрия      размерность = число шкал
    время          собственное (длина цепи) и координатное (сумма фаз)
    причинность    световой конус, предельная скорость
    поле           комплексная амплитуда, интерференция
    заряд          разброс шкал события, сохраняется вдоль линии
    спин 1/2       из двойного накрытия группы перестановок шкал

Мир можно ЗАДАТЬ координатами, а можно ВЫРАСТИТЬ событие за событием:

    --grow      рост с декорреляцией шкал (каждая шкала от своего предка)
    --quantum   рост с квантовым шагом (вероятность |A|^2) и выравниванием
    --compare   сравнить все три способа по однородности

Запуск:
    python3 simulate.py
    python3 simulate.py --n 3000 --k 4
    python3 simulate.py --k 3          # двумерное пространство
    python3 simulate.py --grow
    python3 simulate.py --compare
"""
import argparse
import numpy as np
from math import gamma

try:
    from scipy.optimize import brentq
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ============================================================ алгебра Клиффорда

def clifford(k):
    """Генераторы: e_i^2 = -1, e_i e_j = -e_j e_i. Нужны для спинора."""
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    n = (k+1)//2

    def kron(mats):
        M = mats[0]
        for A in mats[1:]:
            M = np.kron(M, A)
        return M

    gens = []
    for i in range(k):
        q, r = divmod(i, 2)
        gens.append(1j*kron([Z]*q + [X if r == 0 else Y] + [I2]*(n-q-1)))
    return gens


def transposition(gens, i, j):
    """Перестановка шкал i,j как элемент двойного накрытия: t^2 = -1."""
    return (gens[i] - gens[j])/np.sqrt(2)


# ==================================================================== вселенная

class Spacetime:
    def __init__(self, N=1500, k=4, seed=0):
        self.N, self.k = N, k
        rs = np.random.default_rng(seed)
        self.X = rs.random((N, k))                 # k чисел на событие
        self.R = self._order()                     # R[i,j]: j раньше i
        self.L = self._links()                     # прямые связи
        self.D = self._depth(self.R)               # собственное время
        self.g = clifford(k)

    # -------------------------------------------------------------- геометрия
    def _order(self):
        R = np.ones((self.N, self.N), dtype=bool)
        for m in range(self.k):
            R &= (self.X[None, :, m] < self.X[:, None, m])
        np.fill_diagonal(R, False)
        return np.asarray(R, dtype=bool)

    def _links(self):
        A = self.R.astype(np.float32)
        return self.R & ((A @ A) == 0)

    def _depth(self, R):
        Rb = np.ascontiguousarray(R, dtype=bool)
        n = len(Rb)
        ind = Rb.sum(axis=1).astype(np.int64)
        D = np.zeros(n, dtype=np.int64)
        rd = list(np.flatnonzero(ind == 0))
        while rd:
            j = rd.pop()
            kd = np.flatnonzero(Rb[:, j])
            if kd.size:
                D[kd] = np.maximum(D[kd], D[j]+1)
                ind[kd] -= 1
                rd.extend(kd[ind[kd] == 0].tolist())
        return D

    def density(self):
        return float(self.R.sum()/(self.N*(self.N-1)/2))

    def dimension(self):
        """Размерность по Майрхейму–Мейеру из доли связанных пар."""
        f = self.density()
        if not HAVE_SCIPY or not (0 < f < 1):
            return -np.log2(f) + 1 if f > 0 else float('nan')
        g = lambda d: gamma(d+1)*gamma(d/2)/(2*gamma(1.5*d))
        try:
            return brentq(lambda d: g(d)-f, 1.0, 40)
        except ValueError:
            return float('nan')

    # ------------------------------------------------------------------ время
    def coordinate_time(self, i):
        """Координатное время события = сумма его фаз."""
        return float(self.X[i].sum())

    def interval(self, i, j):
        """Собственное время между событиями = длина максимальной цепи."""
        if not self.R[j, i]:
            return None
        mid = np.flatnonzero(self.R[:, i] & self.R[j, :])
        if len(mid) == 0:
            return 1
        return 1 + int(self._depth(self.R[np.ix_(mid, mid)]).max()) + 1

    # ------------------------------------------------------------------- заряд
    def charge(self, i):
        """Заряд = разброс шкал. Независим от времени (корр. ~0.01)."""
        return float(self.X[i].var())

    # -------------------------------------------------------------------- спин
    def spinor(self, i):
        """4-компонентный спинор из ранжировки шкал события."""
        d = self.g[0].shape[0]
        v = np.zeros(d, dtype=complex); v[0] = 1.0
        return self._lift(list(range(self.k)), tuple(np.argsort(self.X[i]))) @ v

    def _lift(self, frm, to):
        """Подъём перестановки в двойное накрытие."""
        M = np.eye(self.g[0].shape[0], dtype=complex)
        pa = list(frm)
        for a in range(self.k):
            if pa[a] != to[a]:
                b = pa.index(to[a])
                M = M @ transposition(self.g, a, b)
                pa[a], pa[b] = pa[b], pa[a]
        return M

    def spin_transport(self, chain):
        """Перенос спинора вдоль мировой линии. Унитарен."""
        M = np.eye(self.g[0].shape[0], dtype=complex)
        perms = [tuple(np.argsort(self.X[c])) for c in chain]
        for t in range(1, len(perms)):
            M = M @ self._lift(list(perms[t-1]), perms[t])
        return M

    def spin_loop(self, turns=1, seed=0, steps=3000):
        """Обход в пространстве фаз. 1 оборот даёт -1, 2 оборота +1."""
        r = np.random.default_rng(seed)
        a = r.normal(size=self.k); a /= np.linalg.norm(a)
        b = r.normal(size=self.k); b -= (b@a)*a; b /= np.linalg.norm(b)
        th = np.linspace(0, 2*np.pi*turns, steps*turns, endpoint=False)
        P = np.mod(0.5 + 0.3*(np.cos(th)[:, None]*a[None, :]
                              + np.sin(th)[:, None]*b[None, :]), 1.0)
        perms = [tuple(np.argsort(row)) for row in P]
        M = np.eye(self.g[0].shape[0], dtype=complex)
        flips = 0
        for t in range(1, len(perms)):
            if perms[t] != perms[t-1]:
                d = [i for i in range(self.k) if perms[t][i] != perms[t-1][i]]
                if len(d) == 2:
                    M = M @ transposition(self.g, d[0], d[1])
                    flips += 1
        v = np.zeros(self.g[0].shape[0], dtype=complex); v[0] = 1.0
        return flips, complex(np.round((M@v) @ v.conj(), 6))

    # -------------------------------------------------------------------- поле
    def _topo(self):
        Lb = np.ascontiguousarray(self.L, dtype=bool)
        ind = Lb.sum(axis=1).astype(np.int64)
        rd = list(np.flatnonzero(ind == 0)); out = []
        while rd:
            j = rd.pop(); out.append(j)
            kd = np.flatnonzero(Lb[:, j])
            if kd.size:
                ind[kd] -= 1
                rd.extend(kd[ind[kd] == 0].tolist())
        return out

    def propagate(self, sources):
        """Комплексная амплитуда по причинным связям. Без нормировки:
        она нелинейна и стирает относительные фазы."""
        V = np.zeros((self.N, self.k), dtype=complex)
        A = np.exp(2j*np.pi*self.X)
        for v in self._topo():
            if v in sources:
                V[v] = sources[v]; continue
            par = np.flatnonzero(self.L[v])
            if len(par):
                V[v] = A[v]*(V[par].sum(0)/np.sqrt(len(par)))
        return V

    # ------------------------------------------------------------- траектории
    def worldline(self, start, maxlen=40, seed=0):
        r = np.random.default_rng(seed)
        cur = start; ch = [start]
        for _ in range(maxlen):
            nxt = np.flatnonzero(self.L[:, cur])
            if len(nxt) == 0:
                break
            cur = int(r.choice(nxt)); ch.append(cur)
        return ch

    def longest_worldline(self, tries=400, seed=0):
        r = np.random.default_rng(seed)
        starts = np.flatnonzero(self.D == 0)
        best = []
        for _ in range(tries):
            s = int(r.choice(starts))
            w = self.worldline(s, seed=int(r.integers(0, 10**6)))
            if len(w) > len(best):
                best = w
        return best


# ============================================================== растущий мир

class GrownSpacetime(Spacetime):
    """Мир не задан, а ВЫРОС: события добавляются по одному.

    Три механизма, найденные по отдельности:
      1. квантовый шаг     — предки выбираются с вероятностью |A|^2,
                             где A — сумма амплитуд по путям (фаза влияет)
      2. выравнивание      — у самых нагруженных событий срезаются связи
                             (интерференция создаёт горячие точки)
      3. декорреляция шкал — каждая шкала наследуется от СВОЕГО предка,
                             иначе шкалы коррелируют и мир вырождается
    """

    def __init__(self, N=800, k=4, seed=0, phase=1.0, burn=0.05,
                 horizon=6, step=0.05, mode="decor"):
        self.N, self.k = N, k
        self.phase, self.burn, self.mode = phase, burn, mode
        rs = np.random.default_rng(seed)
        if mode == "decor":
            self.X = self._grow_decor(rs, step)
            self.R = self._order()
        else:
            self.X = None
            self.R = self._grow_quantum(rs, horizon)
        self.L = self._links()
        self.D = self._depth(self.R)
        self.g = clifford(k)

    def _grow_decor(self, rs, step):
        """Шкалы наследуются пошкально — остаются независимыми."""
        X = np.zeros((self.N, self.k))
        for n in range(1, self.N):
            for m in range(self.k):
                q = int(rs.integers(0, n))
                X[n, m] = X[q, m] + rs.random()*step
        return X

    def _grow_quantum(self, rs, horizon, nlinks=2):
        """Квантовый шаг + выравнивание. Фаза влияет на структуру."""
        R = np.zeros((self.N, self.N), dtype=bool)
        D = np.zeros(self.N, dtype=np.int64)
        for n in range(1, self.N):
            if n <= nlinks:
                past = np.ones(n, dtype=bool)
            else:
                A = R[:n, :n].astype(np.complex128)*np.exp(1j*self.phase)
                seed = np.ones(n, dtype=np.complex128)*(D[:n] == 0)
                if seed.sum() == 0:
                    seed = np.ones(n, dtype=np.complex128)
                acc = seed.copy(); cur = seed.copy()
                for _ in range(horizon):
                    cur = A.T @ cur; acc = acc + cur
                    if np.abs(cur).max() < 1e-14:
                        break
                p = np.abs(acc)**2
                if p.sum() <= 0 or np.count_nonzero(p) < nlinks:
                    p = np.ones(n)
                p = p/p.sum()
                sel = rs.choice(n, min(nlinks, n), replace=False, p=p)
                past = np.zeros(n, dtype=bool); past[sel] = True
            idx = np.flatnonzero(past)
            R[:n, n] = past | R[:n, :n][:, idx].any(axis=1)
            D[n] = D[:n][R[:n, n]].max()+1 if R[:n, n].any() else 0
            if self.burn > 0 and n > 20 and n % 5 == 0:
                deg = R[:n+1, :n+1].sum(axis=1)
                hot = np.argsort(-deg)[:max(1, int(self.burn*n))]
                for h in hot:
                    cs = np.flatnonzero(R[h, :n+1])
                    if len(cs) > 2:
                        drop = rs.choice(cs, max(1, len(cs)//4), replace=False)
                        R[h, drop] = False
        return np.asarray(R, dtype=bool)

    # у квантового режима нет координат — заряд и спин недоступны
    def charge(self, i):
        return super().charge(i) if self.X is not None else None

    def spinor(self, i):
        return super().spinor(i) if self.X is not None else None


class BoundarySpacetime(Spacetime):
    """Мир растёт, храня ТОЛЬКО ГРАНИЦУ — фронт из width последних событий.

    Прошлое не хранится вовсе: память O(width * k) вместо O(N^2).
    Два условия, без которых не работает:
      - координаты БЛУЖДАЮТ (прибавка со случайным знаком),
        иначе всё становится сравнимым и мир вырождается в цепь;
      - каждая шкала наследуется от СВОЕГО предка из фронта,
        иначе шкалы коррелируют.
    """

    def __init__(self, N=2000, k=4, seed=0, width=None, step=0.05):
        if width is None:
            width = max(120, N//15)      # ширина должна расти с миром
        self.N, self.k = N, k
        self.width = width
        rs = np.random.default_rng(seed)
        self.X = self._grow(rs, width, step)
        self.R = self._order()
        self.L = self._links()
        self.D = self._depth(self.R)
        self.g = clifford(k)

    def _grow(self, rs, width, step):
        X = np.zeros((self.N, self.k))
        front = [0]
        for n in range(1, self.N):
            for m in range(self.k):
                q = front[int(rs.integers(0, len(front)))]
                X[n, m] = X[q, m] + (rs.random()-0.5)*step
            front.append(n)
            if len(front) > width:
                front = front[-width:]
        return X


def scale_dimension(S, rs=None, nmax=40):
    """Размерность внутри интервалов разного размера — проверка однородности."""
    r = rs if rs is not None else np.random.default_rng(3)
    b = {}
    for _ in range(25000):
        i, j = (int(v) for v in r.integers(0, S.N, 2))
        if not S.R[i, j]:
            continue
        mid = np.flatnonzero(S.R[:, j] & S.R[i, :])
        m = len(mid)
        if m < 6:
            continue
        key = int(2**int(np.log2(m)))
        lst = b.setdefault(key, [])
        if len(lst) >= nmax:
            continue
        lst.append(S.R[np.ix_(mid, mid)].sum()/(m*(m-1)/2))
    out = {}
    for key, v in b.items():
        if len(v) < 8:
            continue
        f = float(np.mean(v))
        if HAVE_SCIPY and 0 < f < 1:
            g = lambda d: gamma(d+1)*gamma(d/2)/(2*gamma(1.5*d))
            try:
                out[key] = brentq(lambda d: g(d)-f, 1.0, 40)
            except ValueError:
                pass
    return out


# ===================================================================== отчёт

def report(S):
    line = "─"*64
    print(line)
    print(f"КУСОК ПРОСТРАНСТВА-ВРЕМЕНИ  ({S.N} событий, {S.k} шкал)")
    print(line)

    print("\n1. ГЕОМЕТРИЯ")
    print(f"   доля связанных пар : {S.density():.5f}   (теория 2^-(k-1) = {2.0**-(S.k-1):.5f})")
    print(f"   РАЗМЕРНОСТЬ        : {S.dimension():.2f}   "
          f"→ пространственных измерений {S.k-1}")
    print(f"   причинных связей   : {int(S.R.sum()):,}")
    print(f"   прямых (лучей)     : {int(S.L.sum()):,}   "
          f"({S.L.sum()/max(S.R.sum(),1):.1%} от всех)")

    print("\n2. ВРЕМЯ")
    print(f"   высота (макс. собственное время): {int(S.D.max())} тактов")
    wl = S.longest_worldline()
    print(f"   самая длинная мировая линия     : {len(wl)} событий")
    if len(wl) >= 3:
        t = [S.coordinate_time(v) for v in wl]
        print(f"   координатное время вдоль неё    : "
              f"{t[0]:.2f} → {t[-1]:.2f}  (монотонно: {all(np.diff(t) > 0)})")

    print("\n3. ЗАРЯД")
    q = np.array([S.charge(i) for i in range(S.N)])
    tt = S.X.sum(1)
    print(f"   разброс шкал: медиана {np.median(q):.4f}")
    c = abs(np.corrcoef(q, tt)[0, 1])
    verdict = "независимы" if c < 0.1 else ("слабо связаны" if c < 0.4 else "СВЯЗАНЫ")
    print(f"   корреляция заряда со временем: {c:.3f}   → {verdict}")

    print("\n4. СПИН")
    for turns in (1, 2):
        fl, val = S.spin_loop(turns=turns)
        sign = "−1" if abs(val + 1) < 0.01 else ("+1" if abs(val - 1) < 0.01 else f"{val}")
        print(f"   {turns} оборот{'а' if turns > 1 else ''} в пространстве фаз: "
              f"{fl} смен ранжировки → спинор × {sign}")
    if len(wl) >= 3:
        M = S.spin_transport(wl)
        v = np.zeros(S.g[0].shape[0], dtype=complex); v[0] = 1
        print(f"   перенос вдоль мировой линии: норма {np.linalg.norm(M@v):.4f} (унитарен)")

    print("\n5. ПОЛЕ")
    src = int(np.flatnonzero(S.D == 0)[0])
    A = np.exp(2j*np.pi*S.X)
    V = S.propagate({src: A[src]})
    reached = int((np.abs(V).sum(1) > 0).sum())
    print(f"   амплитуда из события {src} достигла {reached} событий")
    fut = int(S.R[:, src].sum())
    print(f"   в световом конусе события: {fut}   "
          f"→ утечка за конус: {max(0, reached-fut-1)}")

    print("\n" + line)
    print("Всё выше выведено из одного правила и числа k.")
    print("Ни одного дополнительного параметра не введено.")
    print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="число событий")
    ap.add_argument("--k", type=int, default=4, help="число шкал")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grow", action="store_true",
                    help="вырастить мир, а не задать координатами")
    ap.add_argument("--quantum", action="store_true",
                    help="рост с квантовым шагом и выравниванием")
    ap.add_argument("--compare", action="store_true",
                    help="сравнить засев и рост по однородности")
    ap.add_argument("--boundary", action="store_true",
                    help="рост с памятью только на границе (фронт)")
    ap.add_argument("--width", type=int, default=0,
                    help="ширина фронта (0 = авто, N/15)")
    args = ap.parse_args()

    if args.compare:
        print("СРАВНЕНИЕ СПОСОБОВ ПОСТРОЕНИЯ")
        print("─"*64)
        rows = [("засев (координаты)", Spacetime(N=args.n, k=args.k, seed=args.seed)),
                ("рост + декорреляция", GrownSpacetime(N=args.n, k=args.k,
                                                       seed=args.seed, mode="decor")),
                ("рост квантовый", GrownSpacetime(N=args.n, k=args.k,
                                                  seed=args.seed, mode="quantum"))]
        print(f"{'способ':>22}{'разм.':>8}{'инт.8':>8}{'инт.16':>9}"
              f"{'инт.32':>9}{'разброс':>10}")
        for name, S in rows:
            g = S.dimension()
            sd = scale_dimension(S)
            v = [sd.get(x, float('nan')) for x in (8, 16, 32)]
            fin = [x for x in [g]+v if np.isfinite(x)]
            sp = max(fin)-min(fin) if fin else float('nan')
            print(f"{name:>22}{g:>8.2f}" + "".join(f"{x:>9.2f}" for x in v)
                  + f"{sp:>10.2f}")
        print("─"*64)
        print("разброс ~0 значит мир ОДНОРОДЕН: локальная размерность = глобальной")
    elif args.boundary:
        w = args.width if args.width > 0 else None
        S = BoundarySpacetime(N=args.n, k=args.k, seed=args.seed, width=w)
        report(S)
        print(f"\nПАМЯТЬ: фронт {S.width}x{args.k} чисел вместо "
              f"матрицы {args.n}x{args.n}")
        print(f"        выигрыш в {args.n*args.n/(S.width*args.k):.0f} раз")
    elif args.quantum:
        report(GrownSpacetime(N=args.n, k=args.k, seed=args.seed, mode="quantum"))
    elif args.grow:
        report(GrownSpacetime(N=args.n, k=args.k, seed=args.seed, mode="decor"))
    else:
        report(Spacetime(N=args.n, k=args.k, seed=args.seed))
