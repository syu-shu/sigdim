# Copyright 2024 Shuriku Kai

# This program is free software: you can redistribute it
# and/or modify it under the terms of the GNU General
# Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version.

# This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License
# for more details.

# You should have received a copy of the GNU General Public
# License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np
from fractions import Fraction
import math
import qsoptex
import logging
import collections

from dual import *
from e2m import *
from vertices import *
from symmetry import *
from sym_ext import *

def sigdim(state: np.ndarray=None, effect:np.ndarray=None, lb:int=2, ub:int=0) -> dict[str, object]:
    if state is None and effect is None:
        return None

    # logging.basicConfig(level=logging.DEBUG)

    # dual
    if effect is None:
        effect = dual(dual_input(state))
        effect = effect[:, 1:]
        effect = effect_lcm(effect)
    elif state is None:
        state = dual(dual_input(effect))
        state = state[:, 1:]
        state = effect_lcm(state)

    is_cs = cs(state[:, :-1])
    if ub == 0:
        ub = state.shape[0]
        if is_cs:
            ub -= 1
    if (not is_cs) and lb == 2:
        lb += 1

    # e2m
    dim = effect.shape[1]
    ext_meass = e2m(e2m_input(effect), dimension=dim)
    ext0 = ext_meass.shape[0]

    # symmetry
    sym = symmetry(effect[:, :-1])
    ext_meass = sym_ext(ext_meass, sym)

    # sigdim
    sigdim = 0
    c = 0
    for ext in ext_meass:
        c += 1
        flags = [x != 0 for x in ext]
        if sum(flags) <= lb:
            continue
        poly = effect[flags]
        poly = ext_lcm(ext, poly)
        P = state @ poly.T
        P = np.unique(P, axis=0)
        # TODO:unique to convexhall
        n = P.shape[1]
        dd = 0
        for d in range(max(lb, sigdim), min(n, ub)):
            A = vertices(P, d)
            if check(P, A):
                dd = d
                break
        else:
            dd = min(n, ub)
        sigdim = max(sigdim, dd)

    return {
        "sigdim":sigdim,
        "cs":is_cs,
        "sym":sym.shape[0],
        "ext0":ext0,
        "ext":ext_meass.shape[0]
    }

def cs(X: np.ndarray) -> bool:
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if np.all(X[i] == -X[j]):
                break
        else:
            return False
    return True



def check(P:np.ndarray, A:np.ndarray) -> bool:
    problem = qsoptex.ExactProblem()
    n = P.shape[1]
    m, v = A.shape
    row = 0
    for col in range(v):
        problem.add_variable(name="x" + str(col), objective=1, lower=0)
    for row_major in range(m):
        for row_minor in range(n):
            constraints = {}
            for col in range(v):
                if row_minor == A[row_major, col]:
                    s = "x" + str(col)
                    constraints[s] = 1
            problem.add_linear_constraint(qsoptex.ConstraintSense.EQUAL, constraints, rhs=int(P[row_major, row_minor]))
    problem.set_objective_sense(qsoptex.ObjectiveSense.MINIMIZE)
    problem.set_param(qsoptex.Parameter.SIMPLEX_DISPLAY, 1)
    status = problem.solve()
    return status == qsoptex.SolutionStatus.OPTIMAL



def ext_lcm(ext: np.ndarray, poly:np.ndarray) -> np.ndarray:
    i = 0
    lcm = 1
    for fac in filter(lambda x: x != 0, ext):
        if type(fac) == Fraction:
            poly[i, :] = poly[i, :] * fac.numerator
            lt = math.lcm(lcm, fac.denominator)
            poly[i, :] = poly[i, :] * (lt // fac.denominator)
            poly[:i, :] = poly[:i, :] * (lt // lcm)
            lcm = lt
        else :
            poly[i] *= fac
        i += 1
    return poly

if __name__ == "__main__":
    state = [
        [1, 0, 0, 1],
        [-1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, -1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, -1, 1]
    ]
    state = np.array(state, dtype=np.int16)
    print(sigdim(state=state, effect=None))