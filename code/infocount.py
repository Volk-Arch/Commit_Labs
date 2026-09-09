import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'minbits.py'), encoding='utf-8').read())

def order_entropy_bits(R):
    """Сколько бит реально несёт порядок: энтропия распределения по различным прошлым."""
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    packed = np.packbits(Rb, axis=1)
    _, cnt = np.unique(packed, axis=0, return_counts=True)
    p = cnt/cnt.sum()
    return float(-(p*np.log2(p)).sum()), len(cnt)

def compressed_bits(R):
    """Оценка сжимаемости матрицы порядка: сколько бит после сжатия."""
    import zlib
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    raw = np.packbits(Rb).tobytes()
    return len(zlib.compress(raw, 9))*8, len(raw)*8
