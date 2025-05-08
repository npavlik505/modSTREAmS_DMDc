# LFT Generation with Named Subsystems (full MIMO)
import numpy as np
import control as ct
from control import ss, tf, interconnect
import os
import pickle

# Directory handling
script_dir = os.path.dirname(__file__)

# utility: static uncertainty block Δ of size rows×cols
def ultidyn(name, size):
    rows, cols = (size, size) if isinstance(size, int) else size
    blk = ss(
        np.zeros((0, 0)),      # no states
        np.zeros((0, cols)),   # no input dynamics
        np.zeros((rows, 0)),   # no output dynamics
        np.eye(rows, cols)     # D = I
    )
    blk.name = name
    return blk

# 1. Load ROM matrices
A = np.load(os.path.join(script_dir, 'A_red_matrix.npy'))
B = np.load(os.path.join(script_dir, 'B_red_matrix.npy'))
# dims: nx = # states = # measurements; nu = # control inputs
nx, nu = A.shape[0], B.shape[1]
p = nx

# 2. Build full-state measurement C, D
C = np.eye(nx)
D = np.zeros((p, nu))

# 3. Nominal ROM system
P_nom = ss(A, B, C, D)
P_nom.name = 'P_nom'

# 4. Static LQR gain K_ss: nu×p mapping
K = np.load(os.path.join(script_dir, 'K.npy'))   # shape nu×p
K_ss = ss(
    np.zeros((0, 0)),
    np.zeros((0, p)),
    np.zeros((nu, 0)),
    K
)
K_ss.name = 'K_ss'

# 5. Weight filter W(s)
with open(os.path.join(script_dir, 'Wmat3_tf.pkl'), 'rb') as f:
    W_scipy = pickle.load(f)
W_ctl = tf(W_scipy.num, W_scipy.den)

# replicate W filter for each measurement channel
W_blocks = []
for i in range(p):
    w = ss(W_ctl)
    w.name = f'W_{i}'
    W_blocks.append(w)

# 6. Uncertainty Δ as p×p static block
DeltaBlock = ultidyn('Delta', p)

# 7. Assemble all subsystems
subsys = [P_nom, K_ss] + W_blocks + [DeltaBlock]

# 8. Build connection list: plant → controller & weights
nconns = []
for i in range(p):
    # P_nom.y[i] feeds K_ss.u[i] and W_i.u[0]
    nconns.append([f'K_ss.u[{i}]',  f'P_nom.y[{i}]'])
    nconns.append([f'W_{i}.u[0]',   f'P_nom.y[{i}]'])
# weight outputs to Δ inputs
for i in range(p):
    nconns.append([f'Delta.u[{i}]', f'W_{i}.y[0]'])
# feedback: sum into plant input P_nom.u[0]
fb = ['P_nom.u[0]', 'K_ss.y[0]'] + [f'Delta.y[{i}]' for i in range(p)]
nconns.append(fb)

# 9. Interconnect into a generalized plant
gen_plant = interconnect(
    subsys,
    connections=nconns,
    inplist=[f'Delta.u[{i}]' for i in range(p)],
    outlist=[f'W_{i}.y[0]'    for i in range(p)]
)

# 10. Convert to StateSpace and form full MIMO LFT
gen_ss = ct.ss(gen_plant)
# verify I/O dims: both should be p
assert gen_ss.ninputs  == p
assert gen_ss.noutputs == p
# upper LFT with p×p Δ block
M_upper = gen_ss.lft(DeltaBlock, ny=p, nu=p)

print(f'LFT generation succeeded. M_upper has {M_upper.noutputs} outputs and {M_upper.ninputs} inputs.')
