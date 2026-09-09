import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/seeded.py').read())

def event_features(U, R):
    """Все характеристики СОБЫТИЯ, какие умеем считать."""
    k, N = U.shape
    Rb = np.ascontiguousarray(R, dtype=bool)
    Rf = np.asarray(R, dtype=np.float32)
    past = Rb.sum(axis=1).astype(float)
    fut = Rb.sum(axis=0).astype(float)
    dep = proper_time(Rf).astype(float)
    codep = proper_time(np.ascontiguousarray(Rf.T)).astype(float)
    C = Rf; btw = (C.T @ C).astype(np.int32)
    lk_in = ((Rf > 0) & (btw.T == 0)).sum(axis=1).astype(float)
    lk_out = ((Rf > 0) & (btw.T == 0)).sum(axis=0).astype(float)
    Rk = U.T.astype(float)/N
    med = np.median(Rk, axis=1)
    dev = np.abs(Rk - med[:, None])
    srt = np.sort(dev, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    spread = Rk.std(axis=1)
    d, ok = directions(U, R)
    dlen = np.linalg.norm(d, axis=1)
    names = ['прошлое', 'будущее', 'глубина', 'ко-глубина', 'связей вниз',
             'связей вверх', 'отрыв метки', 'разброс рангов', 'длина направления',
             'ранг медианный', 'несравнимых']
    F = np.column_stack([past, fut, dep, codep, lk_in, lk_out,
                         margin, spread, dlen, med, N-1-past-fut])
    return F, names

def corr_matrix(F):
    Z = (F - F.mean(0))/np.maximum(F.std(0), 1e-9)
    return (Z.T @ Z)/len(F)

def partial(C, i, j, given):
    """Частная корреляция i и j при фиксированном given."""
    idx = [i, j, given]
    S = C[np.ix_(idx, idx)]
    try: P = np.linalg.inv(S)
    except Exception: return np.nan
    return float(-P[0, 1]/np.sqrt(P[0, 0]*P[1, 1]))

def pair_features(U, R, want=4000, rs=None):
    """Характеристики ПАРЫ событий (связанных и несравнимых)."""
    r = rs if rs is not None else np.random.default_rng(0)
    k, N = U.shape
    Rb = np.ascontiguousarray(R, dtype=bool)
    Rk = U.T.astype(float)/N
    dep = proper_time(np.asarray(R, dtype=np.float32)).astype(float)
    d, ok = directions(U, R)
    rows = []
    C = np.asarray(R, dtype=np.float32); btw = (C.T @ C).astype(np.int32)
    L = Rb & (btw.T == 0)
    tries = 0
    while len(rows) < want and tries < want*40:
        tries += 1
        i, j = (int(v) for v in r.integers(0, N, 2))
        if i == j: continue
        related = bool(Rb[j, i]); rev = bool(Rb[i, j])
        if rev: i, j = j, i; related = True
        pi, pj = Rb[i], Rb[j]
        inter = int((pi & pj).sum())
        szi, szj = int(pi.sum()), int(pj.sum())
        ov = inter/max(min(szi, szj), 1)
        common = np.flatnonzero(pi & pj)
        if len(common) >= 3:
            sub = Rb[np.ix_(common, common)]
            top = int((~sub.any(axis=0)).sum())
            topfrac = top/len(common)
            conflict = top/max(1, min(szi-inter, szj-inter))
        else:
            topfrac = np.nan; conflict = np.nan
        dr = Rk[j] - Rk[i]
        par = np.sign(dr[0]-dr[1])*np.sign(dr[1]-dr[2])*np.sign(dr[0]-dr[2]) if k >= 3 else 0.0
        interval = int((Rb[:, i] & Rb[j, :]).sum()) if related else 0
        rows.append([float(related), abs(dep[i]-dep[j]), ov, topfrac, conflict,
                     float(par), float(interval), float(L[j, i]),
                     float(np.abs(dr).max()-np.abs(dr).min()), float(np.linalg.norm(dr))])
    names = ['связаны', 'разн.глубин', 'перекрытие', 'доля верхушки', 'конфликтность',
             'чётность прир.', 'мощн.интервала', 'прямая связь', 'перекос прир.', 'длина прир.']
    A = np.array(rows, dtype=float)
    good = ~np.isnan(A).any(axis=1)
    return A[good], names

def interval_features(U, R, want=1200, lo=8, hi=400, rs=None):
    """Характеристики ИНТЕРВАЛА между двумя связанными событиями."""
    r = rs if rs is not None else np.random.default_rng(0)
    k, N = U.shape
    Rb = np.ascontiguousarray(R, dtype=bool)
    Rk = U.T.astype(float)/N
    rows = []; tries = 0
    while len(rows) < want and tries < want*60:
        tries += 1
        j = int(r.integers(1, N)); i = int(r.integers(0, N))
        if i == j or not Rb[j, i]: continue
        mid = np.flatnonzero(Rb[:, i] & Rb[j, :])
        m = len(mid)
        if not (lo <= m < hi): continue
        sub = Rb[np.ix_(mid, mid)]
        subf = sub.astype(np.float32)
        h = int(proper_time(subf).max()) + 1
        f = sub.sum()/max(m*(m-1)/2, 1)                    # доля связанных внутри
        btw = (subf.T @ subf).astype(np.int32)
        nlink = int(((subf > 0) & (btw.T == 0)).sum())
        maximal = int((~sub.any(axis=0)).sum())            # ширина верхнего слоя
        minimal = int((~sub.any(axis=1)).sum())
        dr = Rk[j] - Rk[i]
        par = np.sign(dr[0]-dr[1])*np.sign(dr[1]-dr[2])*np.sign(dr[0]-dr[2]) if k >= 3 else 0.0
        vol = float(np.prod(np.abs(dr)))                   # объём в фазах
        rows.append([m, h, f, nlink/max(m, 1), maximal/max(m, 1), minimal/max(m, 1),
                     float(par), vol, float(h)/max(m**(1/3), 1), float(np.abs(dr).std())])
    names = ['мощность', 'высота', 'плотн.внутри', 'связей/элем', 'ширина верха',
             'ширина низа', 'чётность прир.', 'объём в фазах', 'форма h/m^(1/3)',
             'разброс прир.']
    return np.array(rows, dtype=float), names
