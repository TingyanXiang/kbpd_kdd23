from functools import partial

def f(a,b,c):
    return print(a,b,c)

g = partial(f,b=1)
g(5,6)

