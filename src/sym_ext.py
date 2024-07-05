import numpy as np

def sym_ext (EXT: np.ndarray, SYM: np.ndarray, d:int=-1) -> np.ndarray:
    E, N = EXT.shape
    S = SYM.shape[0]
    if d == -1:
        d = SYM.shape[1]
    ext_reduced = []
    for e in range(E):
        ext = EXT[e, :][SYM[0]]
        for s in range(1, S):
            temp = EXT[e, :][SYM[s]]
            if compare(ext, temp) < 0:
                ext = temp
        # bin search
        l = 0
        r = len(ext_reduced) - 1
        idx = 0
        while l <= r:
            idx = (l + r) // 2
            cmp = compare(ext, np.array(ext_reduced[idx]))
            if cmp > 0:
                l = idx + 1
            elif cmp < 0:
                r = idx - 1
            else:
                break
        else:
            ext_reduced.insert(l, ext.tolist())
    return np.array(ext_reduced)

# A > B : 1
# A == B : 0
# A < B : -1
def compare (A: np.ndarray, B: np.ndarray) -> int:
    N = A.shape[0]
    for i in range(N):
        if A[i] > B[i]:
            return 1
        elif A[i] < B[i]:
            return -1
    return 0