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

def sigdim(state: np.ndarray=None, effect:np.ndarray=None, lb:int=2, ub:int=0) -> int:
    if state is None and effect is None:
        return -1

    # logging.basicConfig(level=logging.DEBUG)

    # dual
    if effect is None:
        effect = dual(dual_input(state))
        effect = effect[:, 1:]
    elif state is None:
        state = dual(dual_input(effect))
        print(state)
        state = state[:, 1:]

    if ub == 0:
        ub = state.shape[0]
        if cs(state[:, :-1]):
            ub -= 1
    if (not cs(state[:, :-1])) and lb == 2:
        lb += 1

    # e2m
    print("e2m")
    dim = effect.shape[1]
    effect = effect_lcm(effect)
    ext_meass = e2m(e2m_input(effect), dimension=dim)
    print(ext_meass.shape)
    print(ext_meass)

    # symmetry
    print("symmetry")
    sym = symmetry(effect[:, :-1])
    # print(sym.shape)
    # print(ext_meass.shape)
    ext_meass = sym_ext(ext_meass, sym)
    print(ext_meass.shape)
    # print(ext_meass.shape)
    # print(ext_meass * 240)
    # np.savetxt("ext.dat", ext_meass * 240, fmt="%3d")

    print("sigdim")
    sigdim = 0
    c = 0
    for ext in ext_meass:
        c += 1
        print("---")
        print(c)
        flags = [x != 0 for x in ext]
        if sum(flags) <= lb:
            continue
        poly = effect[flags]
        poly = ext_lcm(ext, poly)
        P = state @ poly.T
        P = np.unique(P, axis=0)
        # TODO:unique to convexhall
        n = P.shape[1]
        print(P.shape)
        # print(f"n={n}")
        # for d in range(lb, n + 1):
        #     A = vertices(P, d)
        #     if check(P, A):
        #         print(f"ext[{i}]:sigdim={d}")
        #         sigdim = max(sigdim, d)
        #         break
        dd = 0
        for d in range(max(lb, sigdim), min(n, ub)):
            A = vertices(P, d)
            print(f"{d}, {A.shape}")
            if check(P, A):
                dd = d
                if c == 4:
                    np.savetxt("P.dat", P, fmt="%d")
                    np.savetxt("A.dat", A, fmt="%d")
                break
        else:
            dd = n
        sigdim = max(sigdim, dd)
        print(f"ext[{c}]:sigdim={dd}")

    return sigdim

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
            # problem.add_linear_constraint(qsoptex.ConstraintSense.EQUAL, constraints, rhs=1 if P[row_major, row_minor] > 0 else 0)
    problem.set_objective_sense(qsoptex.ObjectiveSense.MINIMIZE)
    problem.set_param(qsoptex.Parameter.SIMPLEX_DISPLAY, 1)
    # status = problem.solve()
    # return status == qsoptex.SolutionStatus.OPTIMAL
    return solve(problem) == qsoptex.SolutionStatus.OPTIMAL

def solve(problem):
            # problem.add_linear_constraint(qsoptex.ConstraintSense.EQUAL, constraints, rhs=1 if P[row_major, row_minor] > 0 else 0)
    status = problem.solve()
    return status



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
            # problem.add_linear_constraint(qsoptex.ConstraintSense.EQUAL, constraints, rhs=1 if P[row_major, row_minor] > 0 else 0)
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
    # state = [
    #     [2,0,-2,0,0,0,-0,0,-2,-0,2,-0,0,0,-0,0],
    #     [0,2,0,-2,0,0,0,-0,-0,-2,-0,2,0,0,0,-0],
    #     [2,2,2,2,0,0,0,0,-2,-2,-2,-2,0,0,0,0],
    #     [0,0,-0,0,2,0,-2,0,0,0,-0,0,-2,-0,2,-0],
    #     [0,0,0,-0,0,2,0,-2,0,0,0,-0,-0,-2,-0,2],
    #     [0,0,0,0,2,2,2,2,0,0,0,0,-2,-2,-2,-2],
    #     [2,0,-2,0,2,0,-2,0,2,0,-2,0,2,0,-2,0],
    #     [0,2,0,-2,0,2,0,-2,0,2,0,-2,0,2,0,-2],
    #     [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    # ]
    # print([len(i) for i in state])
    state = np.array(state, dtype=np.int16)
    # e = dual(dual_input(state))
    # print(e)
    # print(dual(dual_input(e[:, 1:])))
    # state = state.T
    # print(cs(state[:, :-1]))
    print(sigdim(state=None, effect=dual(dual_input(state))[:, 1:]))
    # print(state.shape)
    # state = np.concatenate([state, np.ones((state.shape[0], 1), dtype=np.int16)], axis=1)