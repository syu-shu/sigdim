import cdd
import numpy as np
import math

# effect to ext.meas
def e2m(E: np.ndarray, dimension:int=4) -> np.ndarray:
    e2m_mat = cdd.Matrix(E.tolist(), number_type="fraction")
    e2m_mat.rep_type = cdd.RepType.INEQUALITY
    e2m_mat.lin_set = tuple([i for i in range(dimension)])
    poly = cdd.Polyhedron(e2m_mat)
    ext_meas_mat = poly.get_generators()
    ext_meas = np.array([list(x[1:]) for x in ext_meas_mat])
    return ext_meas

def e2m_input(effect: np.ndarray) -> np.ndarray:
    e = np.insert(effect, 0,
        np.zeros(
            (1, effect.shape[1]),
            dtype=np.int16),
        0)
    e[0, -1] = -1
    e = e.T
    under = np.eye(e.shape[1] - 1, dtype=np.int16)
    under = np.concatenate([np.zeros((e.shape[1] - 1, 1), dtype=np.int16), under], 1)
    e = np.concatenate([e, under])
    return e

def effect_lcm(effect: np.ndarray) -> np.ndarray:
    lcm = 1
    m = effect.shape[0]
    for i in range(m):
        temp = math.lcm(lcm, effect[i, -1])
        effect[i] = effect[i] * (temp // effect[i, -1])
        effect[:i] = effect[:i] * (temp // lcm)
        lcm = temp
    return effect

if __name__ == "__main__":
    effect = [[1, -1, 1, 1],
                    [1, -1, -1, 1],
                    [1, 1, -1, 1],
                    [1, 1, 1, 1],
                    [-1, 1, -1, 1],
                    [-1, 1, 1, 1],
                    [-1, -1, -1, 1],
                    [-1, -1, 1, 1]]
    print(e2m(e2m_input(effect_lcm(np.array(effect, dtype=np.int16)))))