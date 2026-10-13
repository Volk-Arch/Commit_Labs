"""Доверие ко всему радарному выводу: работает ли ПОДЛИННАЯ цепь.

Мои наблюдатели были ПОДСАЖЕНЫ — плотные линии по 110-130 событий. Но
подсаженная плотная линия есть лестница (анти-масса, delay-doc.md), то
есть искусственный объект. Настоящая максимальная цепь сприклинга коротка:
её длина идёт как N^0.25, то есть десяток событий при N = 5000.

Вопрос: работает ли радар на ПОДЛИННОЙ цепи, и улучшается ли с ростом N.
Если да — вывод внутренний, а подсадка была лишь удобством.
Если нет — вывод держался на искусственном объекте.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from frames import diamond, rel_of, radar

def longest_chain(R):
    """Самая длинная цепь: ДП по топологическому порядку."""
    n = len(R)
    best = np.ones(n, np.int64); prev = -np.ones(n, np.int64)
    for i in range(n):
        pre = np.flatnonzero(R[:, i])
        if len(pre):
            j = int(pre[np.argmax(best[pre])])
            if best[j] + 1 > best[i]:
                best[i] = best[j] + 1; prev[i] = j
    i = int(np.argmax(best)); out = []
    while i >= 0:
        out.append(i); i = int(prev[i])
    return out[::-1]

print('ПОДЛИННАЯ максимальная цепь как наблюдатель (ничего не подсажено)')
print(f"{'N':>7}{'длина цепи':>12}{'покрытие':>10}{'время':>9}"
      f"{'простр.':>9}{'событий':>9}")
rows = []
for N in (1000, 2000, 4000, 8000):
    ct, cr, ln, cov = [], [], [], []
    for s in range(3):
        X = diamond(N, 1.0, np.random.default_rng(71000 + 29*s))
        R = rel_of(X)
        ch = longest_chain(R)
        del R
        C = X[ch]
        t1, r1, ok = radar(X, N, C)
        if ok.sum() < 30: continue
        tt = X[:, 0]; rr = np.linalg.norm(X[:, 1:], axis=1)
        ct.append(np.corrcoef(t1[ok], tt[ok])[0,1])
        cr.append(np.corrcoef(r1[ok], rr[ok])[0,1])
        ln.append(len(ch)); cov.append(ok.mean())
        del X
    if not ct: print(f'{N:>7}  мало данных'); continue
    rows.append((N, np.mean(ct), np.mean(cr), np.mean(ln)))
    print(f'{N:>7}{np.mean(ln):>12.1f}{np.mean(cov):>10.3f}'
          f'{np.mean(ct):>9.4f}{np.mean(cr):>9.4f}'
          f'{int(np.mean(cov)*N):>9}')
A = np.array(rows)
print()
print(f'  длина цепи ~ N^{np.polyfit(np.log(A[:,0]), np.log(A[:,3]), 1)[0]:+.3f}'
      f'   (ожидание +0.25)')
print(f'  корреляция по времени {A[0,1]:.4f} -> {A[-1,1]:.4f}')
print(f'  корреляция по простр. {A[0,2]:.4f} -> {A[-1,2]:.4f}')
print()
print('  для сравнения, подсаженная плотная линия (130 событий):')
print('    время 0.9999, простр. 0.9997')
print()
print('  растёт с N => вывод ВНУТРЕННИЙ, подсадка была удобством')
