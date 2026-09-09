# Драйверы: как считались таблицы в документах

### Честное предупреждение

Из 174 файлов кода **запускаемых только 6** — остальное библиотеки функций. Вызовы,
которыми считались таблицы в `docs/`, жили в интерпретаторе сессии и в файлы не попали.

Ниже — восстановленные драйверы. **Они не проверены построчно против того, что дало
исходные числа.** Считайте их отправной точкой, а не воспроизведением: параметры могли
отличаться, а часть таблиц собиралась в несколько заходов.

### Что проверено запуском

| драйвер | результат проверки |
|---|---|
| **tradeoff-doc** (кривая размена) | ⭐ размерности **3.69, 4.09, 4.89** против **3.75, 4.11, 4.88** в документе — совпало |
| **propagator-doc** | ⭐ первый шаг **1.00 путей, 0.2500** — точно как в документе |
| threebody.py после дописывания | ⭐ запускается, выдаёт таблицу |

Прогоны на меньших мирах (800 и 400 событий вместо 1200 и 500), поэтому доля внутри и
дальние шаги отличаются — это ожидаемо, не расхождение.

**Остальные драйверы запуском не проверены.**

---

## Запускаемые как есть

```bash
python3 code/simulate.py              # сводный прогон, восемь величин
python3 code/measure_constant.py --n 4000 --seeds 40 --save run.tsv
python3 code/analyze_run.py run.tsv   # разбор сохранённого прогона
python3 code/twobody.py --seeds 5 --gap 0.35
python3 code/threebody.py --seeds 5 --gap 0.45
python3 code/order6.py                # сводка по шкальным конструкциям
```

---

## gravity-doc.md — взаимодействие тел

```python
import sys; sys.path.insert(0,'code')
from measure_constant import local_density, metric, causal_order, field_at
import numpy as np

N, k, alpha, spread = 2500, 4, 1.5, 0.05

def field(bg, bodies, mid, nsample=120):
    X = np.vstack([bg] + bodies)
    m = np.zeros(len(X), bool); m[:len(bg)] = True
    d = local_density(X, 0.2); W = metric(d, k, alpha)
    R = causal_order(X, W)
    return field_at(X, R, mid, m, nsample)

for gap in (0.20, 0.35, 0.50, 0.70):
    vals = []
    for s in range(5):
        rs = np.random.default_rng(s*37)
        bg = rs.random((N, k))
        c1 = np.clip(np.full(k,0.5)-gap/2, 0.05, 0.95)
        c2 = np.clip(np.full(k,0.5)+gap/2, 0.05, 0.95)
        b1 = np.clip(c1+rs.normal(0,spread,(200,k)), 0, 1)
        b2 = np.clip(c2+rs.normal(0,spread,(200,k)), 0, 1)
        mid = (c1+c2)/2
        f2 = field(bg, [b1,b2], mid); f1 = field(bg, [b1], mid)
        if np.isfinite(f2) and np.isfinite(f1): vals.append(f2-f1)
    print(gap, np.mean(vals), np.std(vals)/np.sqrt(len(vals)))
```

Ожидаемые числа: −0.076, −0.095, −0.018, 0.000.

---

## tradeoff-doc.md — кривая размена (главный результат)

```python
import sys; sys.path.insert(0,'code')
from relcut import build_relcut
from cutworld import interior
from bigworld import sprinkle
import numpy as np, math
from scipy.optimize import brentq

def mm(f):
    g = lambda d: math.gamma(d+1)*math.gamma(d/2)/(2*math.gamma(1.5*d))
    return brentq(lambda d: g(d)-f, 1.0, 40)

N, k = 1200, 4
for keep in (1.0, 0.7, 0.5, 0.35, 0.25):
    D, F = [], []
    for s in range(5):
        X = sprinkle(N, k, np.random.default_rng(s*79))
        R, L = build_relcut(X, keep)
        D.append(mm(R.sum()/(N*(N-1)/2)))
        cen = X.mean(0)
        idx = np.argsort(np.linalg.norm(X-cen, axis=1))[:N//2]
        inn = np.zeros(N, bool); inn[idx] = True
        F.append(interior(L, inn)/inn.sum())
    print(keep, np.mean(D), np.mean(F))
```

Ожидаемые: 3.75/0.000, 3.89/0.008, 4.11/0.022, 4.45/0.053, 4.88/0.102.

**Важно:** обрезка по **доле** связей, не по абсолютной длине — иначе результат не
переносится между формами мира (см. раздел 5 в `METHOD.md`).

---

## dynamics-doc.md — аттрактор по размерности

```python
import sys; sys.path.insert(0,'code')
from mixrule import step_thresh
from deathrules import rel_of
import numpy as np

N, k = 700, 4
rs0 = np.random.default_rng(0)
starts = {
 'равномерный': rs0.random((N,k)),
 'два кластера': np.clip(np.where(rs0.integers(0,2,(N,1))==0,0.3,0.7)
                         + rs0.normal(0,0.07,(N,k)), 0, 1),
 'диагональ': np.clip(np.repeat(rs0.random((N,1)),k,axis=1)
                      + rs0.normal(0,0.08,(N,k)), 0, 1),
}
for nm, X0 in starts.items():
    X = X0.copy(); rs = np.random.default_rng(19)
    for it in range(301):
        X = step_thresh(X, 0.02, rs, 0.7, 1.5)
    print(nm, dimension_of(X))   # mm(доля пар), как выше
```

Ожидаемые: 3.77, 3.71, 3.73 — сжатие разброса в 21.7 раза.

---

## propagator-doc.md — пропагатор и интерференция

```python
import sys; sys.path.insert(0,'code')
from propagator import propagator, chain_length, links_of
from mixrule import step_thresh
from deathrules import rel_of
import numpy as np

N, k = 500, 4
X = np.random.default_rng(1).random((N,k))
rs = np.random.default_rng(1)
for it in range(150): X = step_thresh(X, 0.02, rs, 0.7, 1.5)
R = rel_of(X); L = links_of(R)
fwd = float(L.sum(0).mean())
K = propagator(X, R, hop=0.5, maxlen=12)
D = chain_length(R).T          # ⚠ транспонировать: D[a,b] = шагов из a в b
A = L.T.astype(np.float64); P = np.eye(N)
for s in range(1, 5):
    P = P @ A
    m = (D == s)
    print(s, int(m.sum()), float(P[m].mean()), float((np.abs(K[m])**2).mean()))
```

⚠ **Ориентация:** `R[i,j]` значит «$j$ раньше $i$». `chain_length` даёт шаги в обратную
сторону — без транспонирования таблица выйдет пустой.

Ожидаемые: 1 шаг — 1.00 путей, 0.2500; 2 — 2.86, 0.6247; 3 — 5.68, 0.2769.

---

## pathint-doc.md — сумма по историям, когерентность кусков

```python
import sys; sys.path.insert(0,'code')
from bdaction import bd_action
from mixrule import step_thresh
from deathrules import rel_of
import numpy as np

N, k = 800, 4
res = {}
for s in range(3):
    X = np.random.default_rng(s*53).random((N,k))
    rs = np.random.default_rng(s*29+1)
    for it in range(200): X = step_thresh(X, 0.02, rs, 0.7, 1.5)
    R = rel_of(X)
    r2 = np.random.default_rng(s*37+3)
    for _ in range(80000):
        i,j = (int(v) for v in r2.integers(0,N,2))
        if not R[i,j]: continue
        mid = np.flatnonzero(R[:,j] & R[i,:])
        if not (3 <= len(mid) <= 10): continue
        l = res.setdefault(len(mid), [])
        if len(l) >= 200: continue
        l.append(bd_action(np.asarray(R[np.ix_(mid,mid)], dtype=bool), 4))
for v in sorted(res):
    a = np.array(res[v], dtype=float)
    print(v, a.std()/(2*np.pi), abs(np.exp(1j*a).mean()), 1/np.sqrt(len(a)))
```

Ожидаемые: объём 3 — разброс 0.13, когерентность 0.814; объём 10 — 1.47, 0.081.

---

## limits-doc.md — порог расфазировки по размерностям

```python
import sys; sys.path.insert(0,'code')
from threshold import phase_threshold
from bigworld import sprinkle
import numpy as np

for d in (2, 3, 4):
    N = 1500
    P = sprinkle(N, d, np.random.default_rng(d*11))
    dt = P[None,:,0] - P[:,None,0]
    dx2 = ((P[None,:,1:] - P[:,None,1:])**2).sum(-1)
    R = np.asarray((dt > 0) & (dt*dt > dx2), dtype=bool)
    thr, info = phase_threshold(R, np.random.default_rng(7), dim=4)
    print(d, thr)
```

Ожидаемые: 4, 7, 12. При $d$=5 порог напрямую не достигается (интервалов мало),
значение 23 получено экстраполяцией $\text{разброс}\sim\text{объём}^{1.35}$.

---

## gauge-doc.md — холономия

```python
import sys; sys.path.insert(0,'code')
from intrinsic_gauge import link_quantities, holonomy
from holonomy import links_of
from mixrule import step_thresh
from deathrules import rel_of
import numpy as np

N, k = 400, 4
X = np.random.default_rng(1).random((N,k))
rs = np.random.default_rng(1)
for it in range(150): X = step_thresh(X, 0.02, rs, 0.7, 1.5)
R = rel_of(X); L = links_of(R)
Rf = R.astype(np.float32)
cp = Rf @ Rf.T; cf = Rf.T @ Rf
ii, jj = np.nonzero(L)
pm = {(int(a),int(b)): float(v) for a,b,v in zip(ii, jj, cp[ii,jj]-cf[ii,jj])}
h = holonomy(R, L, pm, np.random.default_rng(17))
print(h.mean(), h.std()/np.sqrt(len(h)))
```

Ожидаемые: 22.24 ± 2.54.

⚠ **Контроль обязателен:** «разность прошлого» на том же коде должна дать **ровно 0** —
она телескопирует. Если даёт не ноль, драйвер собран неверно.

---

## Остальные документы

Для них драйверы не восстановлены. Общая схема одинакова:

1. вырастить мир: `step_thresh(X, 0.02, rs, 0.7, 1.5)`, 150–200 шагов;
2. построить порядок: `rel_of(X)`, прямые связи `links_of(R)`;
3. посчитать величину функцией из соответствующего модуля;
4. повторить на 3–5 затравках, вывести среднее и ошибку.

Имя модуля указано в конце каждого документа строкой *«Реализация: `файл.py`»*.

---

## Что осталось несогласованным

| вопрос | статус |
|---|---|
| `threebody.py` был усечён | ⭐ **дописан**, проверен запуском |
| `order2.py`, `order6.py`, `breakit.py` | ⭐ **добавлены в архив** (были только локально) |
| драйверы «моста спин–статистика» | ⊗ не восстановлены |
| `diff -r` архива с папками | см. ниже |

---

## Про `diff -r`

Архив `order-bundle.tar.gz` пересобран сейчас и содержит **177 файлов кода** (было 174) и
37 документов. Добавлены `order2.py`, `order6.py`, `breakit.py`; `threebody.py` дописан.

```bash
mkdir /tmp/chk && tar -xzf order-bundle.tar.gz -C /tmp/chk
diff -r /tmp/chk/bundle/code  ваша_папка/code
diff -r /tmp/chk/bundle/docs  ваша_папка/docs
```

Если у вас архив **старее этой пересборки**, расхождения будут ровно в четырёх названных
файлах.
