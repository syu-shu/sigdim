import numpy as np
import collections

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
            [-1, -1,  3],
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
            [-1, -1,  0],
            [-1, -1,  0],
            [ 1,  0,  1],
            [ 1,  0, -1],
            [-1,  0, -1],
            [-1,  0, -1],
            [ 0,  1,  1],
            [ 0,  1, -1],
            [ 0, -1, -1],
            [ 0, -1, -1],
        ], dtype=np.int32),
        "rhonbic_dodecahedron":
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
        ], dtype=np.int32)
    }
    return dic

# def solids(name:str, dim:int=3, ones:bool=True) -> np.ndarray:
#     A:np.ndarray = None
#     if name == "octahedron":
#         A = hyper_octahedron(dim)
#     elif name == "cube":
#         A = hyper_cube(dim)
#     elif name == "truncated_tetrahedron":
#         A = np.array(
#             [
#                 [ 3,  1,  1],
#                 [ 1,  3,  1],
#                 [ 1,  1,  3],
#                 [-3, -1,  1],
#                 [-1, -3,  1],
#                 [-1, -1,  3],
#                 [-3,  1, -1],
#                 [-1,  3, -1],
#                 [-1, -1,  3],
#                 [ 3, -1, -1],
#                 [ 1, -3, -1],
#                 [ 1, -1, -3],
#             ], dtype=np.int32
#         )
#     elif
#     else :
#         return None
#     if ones:
#         A = cat_ones(A)
#     return A

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