"""Мир на диске: считаем кусками, храним только то, что нужно.

Ключевое: координаты — это ВСЁ состояние мира. Порядок из них выводится.
Значит мир можно писать на диск порциями и обрабатывать по частям.
"""
import numpy as np
import os


def build_to_disk(path, N, d, seed, warp=0.0, chunk=100000, batch=20000):
    """Строим мир порциями, каждую сразу на диск."""
    rs = np.random.default_rng(seed)
    written = 0
    with open(path, 'wb') as f:
        while written < N:
            parts = []; got = 0
            target = min(chunk, N-written)
            while got < target:
                t = rs.uniform(0, 1, batch).astype(np.float32)
                x = rs.uniform(-0.5, 0.5, (batch, d-1)).astype(np.float32)
                r = np.linalg.norm(x, axis=1)
                ok = (r < t) & (r < 1-t)
                if warp:
                    ok &= rs.random(batch) < np.exp(-warp*r)
                if ok.any():
                    parts.append(np.column_stack([t[ok], x[ok]]))
                    got += int(ok.sum())
                del t, x, r, ok
            P = np.vstack(parts)[:target]
            del parts
            f.write(P.tobytes())
            written += len(P)
            del P
    return written


def load_slice(path, d, start, count):
    """Читаем кусок мира с диска."""
    itemsize = 4*d
    with open(path, 'rb') as f:
        f.seek(start*itemsize)
        buf = f.read(count*itemsize)
    return np.frombuffer(buf, dtype=np.float32).reshape(-1, d)


def world_size(path, d):
    return os.path.getsize(path)//(4*d)


def time_window(path, d, t0, t1, chunk=200000):
    """Все события в окне времени [t0, t1) — читаем мир потоком.
    Это и есть 'срез сейчас': прошлое и будущее не нужны."""
    N = world_size(path, d)
    out = []
    for s in range(0, N, chunk):
        P = load_slice(path, d, s, min(chunk, N-s))
        m = (P[:, 0] >= t0) & (P[:, 0] < t1)
        if m.any(): out.append(P[m].copy())
        del P
    return np.vstack(out) if out else np.zeros((0, d), dtype=np.float32)
