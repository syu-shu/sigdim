from sigdim import sigdim
from solids import *

import numpy as np
lb = 3
ub = 8
with open("results/hyper_octahedron_3_7.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["state", "#vertices", "aff.dim", "cs", "#symmetries", "#ext0", "#ext", "sigdim"])
    for d in range(lb, ub):
        state = hyper_octahedron(d)
        ans = sigdim(state=cat_ones(state))
        writer.writerow([
            "hyper_octahedron",
            state.shape[0],
            d,
            ans["cs"],
            ans["sym"],
            ans["ext0"],
            ans["ext"],
            ans["sigdim"]
        ])