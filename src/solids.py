import numpy as np
import collections
import csv

from sigdim import sigdim

def solids() -> dict[str, np.ndarray]:
    dic:dict[str, np.ndarray] = {
        "truncated_tetrahedron":
        np.array([
            [ 3,  1,  1],
            [ 1,  3,  1],
            [ 1,  1,  3],
            [-3, -1,  1],
            [-1, -3,  1],
            [-1, -1,  3],
            [-3,  1, -1],
            [-1,  3, -1],
            [-1,  1, -3],
            [ 3, -1, -1],
            [ 1, -3, -1],
            [ 1, -1, -3],
        ], dtype=np.int32),
        "triakis_tetrahedron":
        np.array([
            [ 5,  5,  5],
            [ 5, -5, -5],
            [-5,  5, -5],
            [-5, -5,  5],
            [-3,  3,  3],
            [ 3, -3,  3],
            [ 3,  3, -3],
            [-3, -3, -3],
        ], dtype=np.int32),
        "cubooctahedron":
        np.array([
            [ 1,  1,  0],
            [ 1, -1,  0],
            [-1,  1,  0],
            [-1, -1,  0],
            [ 1,  0,  1],
            [ 1,  0, -1],
            [-1,  0,  1],
            [-1,  0, -1],
            [ 0,  1,  1],
            [ 0,  1, -1],
            [ 0, -1,  1],
            [ 0, -1, -1],
        ], dtype=np.int32),
        "rhombic_dodecahedron":
        np.array([
            [ 1,  1,  1],
            [-1,  1,  1],
            [ 1, -1,  1],
            [-1, -1,  1],
            [ 1,  1, -1],
            [-1,  1, -1],
            [ 1, -1, -1],
            [-1, -1, -1],
            [ 2,  0,  0],
            [ 0,  2,  0],
            [ 0,  0,  2],
            [-2,  0,  0],
            [ 0, -2,  0],
            [ 0,  0, -2],
        ], dtype=np.int32),
        "truncated_octahedron":
        np.array([
            [ 0,  1,  2],
            [ 0,  1, -2],
            [ 0, -1,  2],
            [ 0, -1, -2],
            [ 1,  0,  2],
            [ 1,  0, -2],
            [-1,  0,  2],
            [-1,  0, -2],
            [ 1,  2,  0],
            [ 1, -2,  0],
            [-1,  2,  0],
            [-1, -2,  0],
            [ 0,  2,  1],
            [ 0,  2, -1],
            [ 0, -2,  1],
            [ 0, -2, -1],
            [ 2,  0,  1],
            [ 2,  0, -1],
            [-2,  0,  1],
            [-2,  0, -1],
            [ 2,  1,  0],
            [ 2, -1,  0],
            [-2,  1,  0],
            [-2, -1,  0],
        ], dtype=np.int32),
        "tetrakis_hexahedron":
        np.array([
            [ 3,  0,  0],
            [ 0,  3,  0],
            [ 0,  0,  3],
            [-3,  0,  0],
            [ 0, -3,  0],
            [ 0,  0, -3],
            [ 2,  2,  2],
            [-2,  2,  2],
            [ 2, -2,  2],
            [-2, -2,  2],
            [ 2,  2, -2],
            [-2,  2, -2],
            [ 2, -2, -2],
            [-2, -2, -2]
        ], dtype=np.int32)
    }
    return dic

def cat_ones(A:np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return np.concatenate([A, np.ones((n, 1), dtype=np.int32)], axis=1)

def hyper_cube(dim:int) -> np.ndarray:
    A = collections.deque([])
    for i in range(1 << dim):
        A.append([(-1) ** ((i >> j) & 1) for j in range(dim)])
    A = np.array(A, dtype=np.int32)
    return A

def hyper_octahedron(dim:int) -> np.ndarray:
    A = np.concatenate([np.eye(dim, dtype=np.int16), -np.eye(dim, dtype=np.int16)])
    return A

if __name__ == "__main__":
    dic = solids()
    with open("results/solids_dual.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "#vertices", "aff.dim", "cs", "#symmetries", "#ext0", "#ext", "sigdim"])
        for name, state in dic.items():
            print(name)
            ans = sigdim(state=cat_ones(state))
            print(ans)
            writer.writerow([
                name,
                state.shape[0],
                state.shape[1],
                ans["cs"],
                ans["sym"],
                ans["ext0"],
                ans["ext"],
                ans["sigdim"]
            ])
            ad = sigdim(effect=cat_ones(state))
            print(ad)
            writer.writerow([
                name + "_dual",
                state.shape[0],
                state.shape[1],
                ad["cs"],
                ad["sym"],
                ad["ext0"],
                ad["ext"],
                ad["sigdim"]
            ])
