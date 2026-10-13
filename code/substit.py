"""Монетка → правило подстановки. Каждое событие детерминированно разворачивается
в несколько (a→…, b→…). Отклонение счёта D(L) = #a(первые L) − f·L растёт как
L^h, h = ln|λ2|/ln λ1 (λ — собственные числа матрицы подстановки).
h = 1/2 ⟺ |λ2| = √λ1: то, что монетка Соркина давала случайно."""
import numpy as np, math
def build(rule, n):
    s = np.array([0], np.int8)
    parts = [np.array(r, np.int8) for r in rule]
    while len(s) < n:
        s = np.concatenate([parts[c] for c in s])
    return s[:n]
RULES = {
  'Фибоначчи  a→ab, b→a           ': ([0, 1], [0]),
  'λ=4,−2     a→aabbbb, b→aa      ': ([0, 0, 1, 1, 1, 1], [0, 0]),
  'λ=4,−2 пер. a→abbabb, b→aa     ': ([0, 1, 1, 0, 1, 1], [0, 0]),
  'λ=5,−3     a→abbbb, b→baaaa    ': ([0, 1, 1, 1, 1], [1, 0, 0, 0, 0]),
  'ЗОЛОТАЯ a→acd b→cd c→abc d→ab  ': ([0, 2, 3], [2, 3], [0, 1, 2], [0, 1]),
}
if __name__ == '__main__':
    N = 4_000_000
    Ls = np.unique(np.logspace(1, math.log10(N) - 0.01, 60).astype(int))
    for name, rule in list(RULES.items()) + [('монетка (Соркин)                ', None)]:
        if rule is None:
            s = (np.random.default_rng(1).random(N) < 0.5).astype(np.int8); lam = ''
        else:
            s = build(rule, N)
            k = len(rule); M = np.array([[rule[j].count(i) for j in range(k)] for i in range(k)])
            ev = sorted(np.linalg.eigvals(M), key=lambda x: -abs(x)); lam = f'h_теор = {math.log(abs(ev[1]))/math.log(abs(ev[0])):+.2f}' if abs(ev[1]) > 1 else 'h_теор = 0 (ограничено)'
        f = np.mean(s == 0)
        D = np.cumsum(s == 0) - f * np.arange(1, N + 1)
        # размах отклонения на масштабе L: max|D| по окну [L/2, L]
        amp = np.array([np.max(np.abs(D[L // 2:L])) for L in Ls])
        sl = np.polyfit(np.log(Ls[10:]), np.log(amp[10:] + 1e-9), 1)[0]
        # «спокойствие»: смены знака D на одну октаву (log2 L)
        sgn = np.sign(D[999:]); flips = np.sum(sgn[1:] != sgn[:-1]) / math.log2(N / 1000)
        print(f'{name} {lam:22s} измерено h = {sl:+.2f};  смен знака D на октаву: {flips:8.1f}')
