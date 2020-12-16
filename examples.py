"""How to use "homsolve()" to solve systems of nonlinear equations.


HomCont:
Solving systems of nonlinear equations by homotopy continuation.

Copyright (C) 2018  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it
and/or modify it under the terms of the MIT License.
"""


import numpy as np
from scipy.optimize import root

from homcont.main import homsolve
from homcont.path import HomPath


# %% 2x2 system of equations

def F(x):
    return np.array([x[0]**3 - 2*x[1] + 2,
                     x[1] - x[0]])


# %% benchmark: Newton's method

# fail for nonnegative initial guess
print(root(F, np.array([0, 0])))

# success for accurate initial guess
print(root(F, np.array([-1, -1])))
# array([-1.76929235, -1.76929235])


# %% define auxiliary system: same dimension, but trivial solution

def G(x):
    return np.array([x[0], x[1]])


# trivial solution:
x0 = np.array([0, 0])
assert np.allclose(G(x0), np.zeros(2))


# %% now: homotopy continuation

# define homotopy function s.t.
# H(x, t=0) = G(x) with trivial solution
# H(x, t=1) = F(x) with solution of interest

def H(y):
    x = y[:-1]
    t = y[-1]
    return t*F(x) + (1-t)*G(x)


# %% run homsolve

# starting point
y0 = np.array([0, 0, 0])

# target value of homotopy parameter
t_target = 1

# initialize hompath instance to log steps of solver
hompath = HomPath(dim=2)

# solve
res = homsolve(H, y0, t_target=t_target, hompath=hompath, verbose=True)


# %% result

print(res)

y1 = res['y']
# array([-1.76929231, -1.76929225,  0.99999995])

x1 = y1[:-1]
# array([-1.76929231, -1.76929225])

hompath.plot()
