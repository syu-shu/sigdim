# Copyright 2024 Shuriku Kai

# This program is free software: you can redistribute it
# and/or modify it under the terms of the GNU General
# Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version.

# This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License
# for more details.

# You should have received a copy of the GNU General Public
# License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

from sigdim_with_c import sigdim
from solids import *

import numpy as np
lb = 3
ub = 7
with open("results/hyper_octahedron_" + str(lb) + "_" + str(ub - 1) + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["state", "#vertices", "aff.dim", "cs", "#symmetries", "#ext0", "#ext", "sigdim"])
    for d in range(lb, ub):
        print(d)
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