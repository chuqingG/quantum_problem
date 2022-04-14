from utils import *

import numpy as np
from enum import Enum
from math import acos, asin, log, atan2
from numpy import exp, sin, cos, sqrt
from scipy.linalg import expm, det

USE_PHASE=False

class DeType(Enum):
    two_single: 1


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

class Gate():
    def __init__(self, value: np.ndarray, name: str=None, tbit=0, cbit=None, params=None):
        self.__value = value
        self.__name = name
        self.__params = params
        self.__tbit = tbit
        self.__cbit = cbit
        self.decomps = []
    
    def __str__(self):
        str_to_print = self.name
        if self.params:
            str_to_print += "("
            for idx in self.params:
                str_to_print += "%.2f," % self.params[idx]
            str_to_print = str_to_print[:-1] + ")"
        return str_to_print

    def print_gates(self):
        '''
        print seq: h0, q0, t0, h1, q1, t1,
        '''
        if not self.decomps:
            error_out("the gate hasn't been decomposed")
        else:
            h0 = "   "
            q0 = "q0:"
            t0 = "   "
            h1 = "   "
            q1 = "q1:"
            t1 = "   "
            for g in self.decomps:
                str = g.str()
                if g.tbit == 0:
                    if g.name == 'CNOT':
                        h0 += "     "
                        q0 += "──o──"
                        t0 += "  │  "
                        h1 += "  │  "
                        q1 += "─(+)─"
                        t1 += "     "
                    else:
                        h0 += "┌" + "─" * (len(str) + 2) + "┐"
                        q0 += "┤ " + str + ' ├'
                        t0 += "└" + "─" * (len(str) + 2) + "┘"
                        h1 += " " * (len(str) + 4)
                        q1 += "─" * (len(str) + 4)
                        t1 += " " * (len(str) + 4)
                else:
                    h0 += " " * (len(str) + 4)
                    q0 += "─" * (len(str) + 4)
                    t0 += " " * (len(str) + 4)
                    h1 += "┌" + "─" * (len(str) + 2) + "┐"
                    q1 += "┤ " + str + ' ├'
                    t1 += "└" + "─" * (len(str) + 2) + "┘"
            for s in [h0, q0, t0, h1, q1, t1]:
                print(s)

    @property
    def value(self):
        return self.__value
    
    @property
    def name(self):
        return self.__name

    @property
    def tbit(self):
        return self.__tbit

    @property
    def cbit(self):
        return self.__cbit

    @property
    def params(self):
        return self.__params
    
    @property
    def str(self):
        return self.__str__

    def column(self, col):
        '''
        return a column as a matrix
        '''
        return self.value[:, col][:, None]

    def is_swap(self):
        return is_swap(self.value)
    
    def is_cnot(self):
        cnot_m = np.eye(4)[:, [0,1,3,2]]
        return np.allclose(cnot_m, self.value)

    def is_control_U(self):
        # FIXME(chuqing): Based on the article and wikipedia, 
        # CU gates just include controlling over 1st qubit?
        return is_control_U(self.value)

    def decompose(self, option: str="default"):
        gb = GateBuilder()
        if option == 'two_single':
            matrixs = product_decomp_SVD(self.value)
            gates = [gb.arr_to_u3(m, i) for i, m in enumerate(matrixs)]
            if None in gates:
                warning_out("Cannot be decomposed to two 1-qubit gates")
            elif not kron_decomp_check(*matrixs, self.value):
                warning_out("Only approximate decomposition")
            # else:
            #     gates_out(gates)
        elif option == 'control_u' and self.is_control_U():
            gates = cu_decomp(self.value)
        else:
            gates = general_decomp(self.value)
            # error_out("Unsupported decomposition type")
        self.decomps += gates


class GateBuilder():
    '''
    A builder for some basic gate
    '''
    def __init__(self):
        self.X = Gate(np.array([[0, 1],
                                [1, 0]]), 'X')
        self.Y = Gate(np.array([[0, -1j],
                                [1j, 0]]), 'Y')
        self.Z = Gate(np.array([[1, 0],
                                    [0, -1]]), 'Z')
        self.H = Gate(np.array([[1, 1],
                                    [1, -1]] / np.sqrt(2)), 'H')
        self.I = Gate(np.eye(2), 'I')
        self.S = Gate(np.array([[1, 0],
                                    [0, 1j]]), 'S')
        self.T = Gate(np.diag([1, np.exp(np.pi * 1j / 4)]), 'T')
        self.CNOT = Gate(np.eye(4)[:,[0, 1, 3, 2]], cbit=0, name='CNOT')
    
    def Rx(self, theta):
        value = expm(-1j * theta / 2 * self.X.value)
        return Gate(value, 'Rx', params={'theta': theta})

    def Ry(self, theta):
        value = expm(-1j * theta / 2 * self.Y.value)
        return Gate(value, 'Ry', params={'theta': theta})

    def Rz(self, theta, tbit=0):
        value = expm(-1j * theta / 2 * self.Z.value)
        return Gate(value, 'Rz', tbit, params={'theta': theta})

    def U3(self, theta, phi, lamb, tbit=0):
        value = np.array([[cos(theta / 2), 
                           -exp(1j * lamb) * sin(theta / 2)],
                          [exp(1j * phi) * sin(theta / 2), 
                           exp(1j * (phi + lamb)) * cos(theta / 2)]])
        return Gate(value, 'U3', tbit, params={'theta': theta,
                                  'phi': phi,
                                  'lambda': lamb})
            
    