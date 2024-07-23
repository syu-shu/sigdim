import cdd
import numpy as np

# get input polyhedron's dual
def dual(S:np.ndarray) -> np.ndarray:
    state_mat = cdd.Matrix(S.tolist(), number_type="fraction")
    state_mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(state_mat)
    effect_mat = poly.get_generators()
    E = np.array([list(x) for x in effect_mat], dtype=np.int16)
    return E

# polytope to dual input
# concatenate zeros(n, 1) left of state
def dual_input(state: np.ndarray) -> np.ndarray:
    m = state.shape[0]
    s = np.concatenate([np.zeros((m, 1)), state], axis=1)
    return s

if __name__ == "__main__" :
    state = [
        [1, 0, 0, 1],
        [-1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, -1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, -1, 1]
    ]
    print(dual(dual_input(np.array(state, dtype=np.int16))))