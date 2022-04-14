import numpy as np
from math import sqrt, acos, exp, asin, log, sin, atan2
from scipy.linalg import expm, det
from itertools import permutations

USE_PHASE=False

def warning_out(info: str):
    print("\033[94m%s\033[0m"%"[Warn]", end=' ')
    print(info)


def error_out(info: str):
    print("\033[95m%s\033[0m"%"[Err]", end=' ')
    print(info)


def gates_out(gates):
    print("\033[93m%s\033[0m"%"[Result]")
    for g in gates:
        print("\t", end='')
        print(g)


def generate_input():
    input_list = []
    bases = np.eye(4)
    for base in permutations(bases):
        input_list.append(np.vstack(base))

    return input_list


def normalization(X: np.ndarray):
    d = abs(det(X))
    print(d)
    return X / sqrt(d)


def is_equal(a, b, cutdown_able=True):
        # or just np.allclose
    if cutdown_able:
        return np.allclose(a, b)
    else:
        return np.all(np.equal(a, b))

def is_equiv(a, b):
    '''
    remove the influence of phase
    '''
    a_idx = np.flatnonzero(a.round(10))
    b_idx = np.flatnonzero(b.round(10))
    if np.allclose(a_idx, b_idx):
        k = a[a_idx] / b[b_idx]
        return np.allclose( k / k[0], np.ones(len(k)))
    else: 
        return False

def is_same(x, y):
    if USE_PHASE:
        return is_equiv(x, y)
    else:
        return is_equal(x, y)

def is_unitray(x):
    n = x.shape[0]
    return np.allclose(x @ x.T.conj(), np.eye(n))


def is_swap(x):
    swap_m = np.eye(4)[:, [0,2,1,3]]
    return is_same(x, swap_m)


def is_control_U(x):
    '''
    Return the control-bit or None
    '''
    if x.shape != (4,4):
        return None
    elif is_same(x[:2, :2], np.eye(2)) and is_unitray(x[2:, 2:]):
        return 1
    elif is_same(x[2:, 2:], np.eye(2)) and is_unitray(x[:2, :2]):
        return 2
    else:
        return None


def get_phase(U):
    """
    Compute the alpha of a q-gate
    """
    n = U.shape[0]
    alpha = np.angle(det(U) ** (1 / n))
    return alpha


def remove_glob_phase(U):
    """
    Remove the global phase of a d*d unitary matrix
    :math:`U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)`
    That is, remove :math:`e^{i\alpha}`
    """
    alpha = get_phase(U)
    return U * np.exp(- 1j * alpha)