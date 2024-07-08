from sigdim import sigdim
from solids import *

import numpy as np
lb = 3
ub = 8
with open("results/hyper_octahedron.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["state", "#vertices", "aff.dim", "cs", "#symmetries", "sigdim"])
    for d in range(lb, ub):
        state = hyper_octahedron(d)
        ans = sigdim(state=cat_ones(state))
        writer.writerow([
            "hyper_octahedron",
            state.shape[0],
            d,
            ans["cs"],
            ans["sym"],
            ans["sigdim"]
        ])