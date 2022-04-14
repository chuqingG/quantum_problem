import numpy as np
from gates import *
from utils import *
from scipy.stats import unitary_group


np.set_printoptions(suppress=True)


print("===================Test 3=====================")
# construct a control-u
b = np.eye(4) * (1+0j)
a = unitary_group.rvs(2)
b[2:, 2:] = a

# decompose and print
g = Gate(b, 'G')
g.decompose('control_u')
g.print_gates()

print("===================Test 4=====================")
a = unitary_group.rvs(4)
g = Gate(a, 'G')
g.decompose()
g.print_gates()
