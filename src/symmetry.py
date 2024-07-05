import numpy as np
import collections

def symmetry (A: np.ndarray, d:int=-1) -> np.ndarray:
    N = A.shape[0]
    if d == -1:
        d = A.shape[1]

    # concat index column
    A = np.concatenate([np.arange(N).reshape(N, 1), A], 1)
    # choice linear independet first d vectors
    p = li(A[:, 1:], d)
    A = np.concatenate([A[p], np.delete(A, p, 0)], axis=0)
    # sort by inner product with first d rows
    A[d:, :] = A[sort_ip(A[:, 1:], d) + d]
    # culc G matrix
    GA = [A[0:i + 1, 1:] @ A[i, 1:] for i in range(N)]

    # dfs
    indexes = collections.deque(list(range(N)))
    permutation = collections.deque([])
    symmetries = collections.deque([])
    def recfun (i=0):
        nonlocal permutation
        nonlocal symmetries
        nonlocal indexes
        if i < d:
            for j in range(N - i):
                temp = indexes.popleft()
                permutation.append(temp)
                ip = A[permutation][:, 1:] @ A[temp, 1:]
                if np.all(GA[i] == ip):
                    recfun(i + 1)
                indexes.append(permutation.pop())
        elif len(permutation) == d:
            B = np.concatenate([A[permutation], np.delete(A, permutation, 0)], axis=0)
            idx = sort_ip(B[:, 1:], d)
            B[d:, :] = B[idx + d]
            GB = [B[0:j + 1, 1:] @ B[j, 1:] for j in range(d, N)]
            l = len(GB)
            for j in range(l):
                if not np.all(GA[d + j] == GB[j]):
                    return
            symmetries.append(B[:, 0].tolist())
    recfun()
    p = np.array(symmetries, np.int32)
    p = p[:, np.argsort(p[0])]
    return p

def sort_ip (A: np.ndarray, d) -> np.ndarray:
    N = A.shape[0]
    # print(A)
    ip = A[0:d, :] @ A[d:, :].T
    ip = np.concatenate([np.arange(N - d).reshape(1, N - d) , ip], axis=0)
    for i in range(ip.shape[0] - 1, 0, -1):
        ip = ip[:, np.argsort(ip[i], kind='stable')]
    return ip[0]

def li (A: np.ndarray, d:int) -> np.ndarray:
    p = collections.deque([])
    p.append(0)
    N = A.shape[0]
    i = 1
    while len(p) < d:
        p.append(i)
        if abs(determinant(A[p] @ A[p].T)) == 0:
            p.pop()
        i += 1
    return np.array(p, dtype=np.int32)

def determinant (X: np.ndarray) -> int:
    n = X.shape[0]
    mul = X[:]
    for i in range(n - 1):
        mul = bird(mul) @ X
    return (-1 + 2 * (n % 2)) * mul[0, 0]

def bird (X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    diagonal = np.trace(X)
    muX = np.zeros(X.shape, dtype=np.int32)
    for i in range(n):
        diagonal -= X[i, i]
        muX[i, i] = -diagonal
    for i in range(n - 1):
        muX[i, i + 1:] = X[i, i + 1:]
    return muX


if __name__ == "__main__":
    effect = []
    n = 3
    for i in range(1 << n):
        effect.append([(-1) ** ((i >> j) & 1) for j in range(n)])

    A = np.array(effect, dtype=np.int32)
    # state = [
    #     [2,0,-2,0,0,0,-0,0,-2,-0,2,-0,0,0,-0,0],
    #     [0,2,0,-2,0,0,0,-0,-0,-2,-0,2,0,0,0,-0],
    #     [2,2,2,2,0,0,0,0,-2,-2,-2,-2,0,0,0,0],
    #     [0,0,-0,0,2,0,-2,0,0,0,-0,0,-2,-0,2,-0],
    #     [0,0,0,-0,0,2,0,-2,0,0,0,-0,-0,-2,-0,2],
    #     [0,0,0,0,2,2,2,2,0,0,0,0,-2,-2,-2,-2],
    #     [2,0,-2,0,2,0,-2,0,2,0,-2,0,2,0,-2,0],
    #     [0,2,0,-2,0,2,0,-2,0,2,0,-2,0,2,0,-2],
    #     # [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    # ]
    # A = np.array(state, dtype=np.int32).T
    # n = 8
    sym = symmetry(A)
    print(sym)
    print(sym.shape)