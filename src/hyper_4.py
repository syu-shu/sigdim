import numpy as np
from solids import *

S = np.array([
    [ 1,  1,  0,  0],
    [ 1, -1,  0,  0],
    [-1,  1,  0,  0],
    [-1, -1,  0,  0],

    [ 1,  0,  1,  0],
    [ 1,  0, -1,  0],
    [-1,  0,  1,  0],
    [-1,  0, -1,  0],

    [ 1,  0,  0,  1],
    [ 1,  0,  0, -1],
    [-1,  0,  0,  1],
    [-1,  0,  0, -1],

    [ 0,  0,  1,  1],
    [ 0,  0,  1, -1],
    [ 0,  0, -1,  1],
    [ 0,  0, -1, -1],

    [ 0,  1,  0,  1],
    [ 0,  1,  0, -1],
    [ 0, -1,  0,  1],
    [ 0, -1,  0, -1],

    [ 0,  1,  1,  0],
    [ 0,  1, -1,  0],
    [ 0, -1,  1,  0],
    [ 0, -1, -1,  0],
])

state = cat_ones(S)
with open("results/hyper_4.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["state", "#vertices", "aff.dim", "cs", "#symmetries", "#ext0", "#ext", "sigdim"])
    ans = sigdim(state)
    writer.writerow([
        "hyper-4-24",
        state.shape[0],
        state.shape[1],
        ans["cs"],
        ans["sym"],
        ans["ext0"],
        ans["ext"],
        ans["sigdim"]
    ])
