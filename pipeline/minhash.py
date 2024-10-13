import numpy as np
import re
from numba import vectorize, guvectorize, uint64


P = 61
Pm = (1<<P) - 1
H = 32
Hm = (1<<H) - 1

_1 = np.uint64(1)
_0 = np.uint64(0)
_P = np.uint64(P)
_Pm = np.uint64(Pm)
_H = np.uint64(H)
_Hm = np.uint64(Hm)

NON_CHAR = re.compile('[^\w ]')


def normalize_for_minhash(txt):
    return NON_CHAR.sub('-', txt.lower())


@vectorize([uint64(uint64, uint64, uint64)], nopython=True)
def affine_61(a, b, x):
    a1, a2 = a & _Hm, a >> _H
    x1, x2 = x & _Hm, x >> _H

    tmp = a1 * x1
    ret = b + (tmp & _Pm) + (tmp >> _P)

    tmp = a1 * x2 + a2 * x1
    ret += ((tmp << H) & _Pm) + (tmp >> (_P - _H))

    tmp = a2 * x2
    ret += ((tmp << 3) & _Pm) + (tmp >> (_P - 3))
    
    return ret % _Pm


@guvectorize([(uint64[:], uint64[:])], '(s)->()')
def sum_61(a, out):
    s = a.shape[0]
    out[0] = np.uint64(0)
    for i in range(s):
        out[0] += a[i]
        out[0] %= _Pm


@guvectorize([(uint64, uint64, uint64[:], uint64[:])], '(),(),(h)->()', nopython=True)
def hashmin_gu(a, b, h, out):
    out[0] = _Pm
    for hi in h:
        out[0] = min(out[0], affine_61(a, b, hi))


def hashit(a, b, z, N, vals):
    acc = vals
    zp = z
    for i in range(0, N):
        d = (1<<i)
        acc = affine_61(zp, acc[:-d], acc[d:])
        zp = affine_61(zp, 0, zp)
    return hashmin_gu(a, b, acc)

