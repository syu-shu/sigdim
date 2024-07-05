from sigdim import *

state = [
    [2,0,-2,0,0,0,-0,0,-2,-0,2,-0,0,0,-0,0],
    [0,2,0,-2,0,0,0,-0,-0,-2,-0,2,0,0,0,-0],
    [2,2,2,2,0,0,0,0,-2,-2,-2,-2,0,0,0,0],
    [0,0,-0,0,2,0,-2,0,0,0,-0,0,-2,-0,2,-0],
    [0,0,0,-0,0,2,0,-2,0,0,0,-0,-0,-2,-0,2],
    [0,0,0,0,2,2,2,2,0,0,0,0,-2,-2,-2,-2],
    [2,0,-2,0,2,0,-2,0,2,0,-2,0,2,0,-2,0],
    [0,2,0,-2,0,2,0,-2,0,2,0,-2,0,2,0,-2],
    [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
]
# print([len(i) for i in state])
state = np.array(state, dtype=np.int16)
state = state.T
# print(state.shape)
# state = np.concatenate([state, np.ones((state.shape[0], 1), dtype=np.int16)], axis=1)
# print(state.shape)
# print(sigdim(state))
effect = dual(dual_input(state))
effect = effect[:, 1:]
print(effect.shape)
print(effect)
dim = effect.shape[1]
ext_meass = e2m(e2m_input(effect), dimension=dim)
print(ext_meass.shape)
print(ext_meass[:11, :] * 240)
np.savetxt("ext.dat", ext_meass * 240, fmt="%d")
# print(sigdim(state))