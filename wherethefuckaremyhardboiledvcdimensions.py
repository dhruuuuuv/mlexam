import operator as op
import math


ds = [2, 3, 5, 10]
qs = [2, 3, 5, 10]

def ncr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

for d in ds:
    for q in qs:
        print("dimensionality of z with d: {}, q: {} = {} == {}".format(d, q, (ncr(d+q,d) <= d ** q), (ncr(d+q, q) <= q ** d)))
