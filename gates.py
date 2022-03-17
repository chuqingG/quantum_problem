from torch import linalg
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

    def decompose(self, option: str):
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
            error_out("Unsupported decomposition type")
            return 
        self.decomps += gates




# Functions for decomposition

def get_u3_params(u):
    if u.shape != (2,2):
        raise ValueError("please provide a 1-qubit gate")
    k = 1 / sqrt(det(u))
    v = (u * k).round(10)
    b_d_sum = 2 * np.angle(v[1, 1])
    b_d_diff = 2 * np.angle(v[1, 0])
    a = - np.angle(k)
    b = (b_d_sum + b_d_diff) / 2
    c = 2 * atan2(abs(v[1,0]), abs(v[0,0]))
    d = (b_d_sum - b_d_diff) / 2

    return a, b, c, d


def product_decomp_SVD(X: np.ndarray, m1=2, n1=2):
    '''
    An approximation algorithm using SVD, the accuracy is bad
    See the algorithm here (p14-19): https://www.imageclef.org/system/files/CLEF2016_Kronecker_Decomposition.pdf
    '''
    m, n = X.shape
    m2, n2 = m // m1, n // n1
    X_split = [X[i*m2:(i+1)*m2, j*n2:(j+1)*n2] for i in range(m1) for j in range(n1)]
    X_flatten = [m.ravel() for m in X_split]
    X_reordered = np.vstack(X_flatten).T
   
    U, S, Vt = np.linalg.svd(X_reordered)
   
    A_re = sqrt(np.max(S)) * U[:,0]
    B_re = sqrt(np.max(S)) * Vt.T[:,0]
    
    A = B_re.reshape(m1, n1).round(10)
    B = A_re.reshape(m2, n2).round(10)
    return A, B


def count_coeff(U, V):
    """
    Claculate the coefficient a, s.t. U = a V
    """
    assert U.shape == V.shape, "input matrices should have the same dimension"
    idx1 = np.flatnonzero(U.round(6))  # cut to some precision
    idx2 = np.flatnonzero(V.round(6))
    try:
        if np.allclose(idx1, idx2):
            return U.ravel()[idx1[0]] / V.ravel()[idx2[0]]
    except:
        return None


def product_decomp_noSVD(X, m1=2, n1=2):
    """
    A manual approximation algorithm without SVD, the accuracy is worse
    """
    m, n = X.shape
    m2, n2 = m // m1, n // n1
    X_split = [X[i*m2:(i+1)*m2, j*n2:(j+1)*n2] for i in range(m1) for j in range(n1)]
    X_flatten = [m.ravel() for m in X_split]
    X_reordered = np.vstack(X_flatten)

    l = [not np.allclose(np.zeros(m), X_reordered[i]) for i in range(4)]

    B_re = X_reordered[l.index(True)]

    A_re = np.array([count_coeff(X_reordered[i], B_re) 
                        if l[i] else 0 for i in range(m)])
    
    A = A_re.reshape(m1, n1)
    B = B_re.reshape(m2, n2)
    return A, B
    

def kron_decomp_check(A, B, X):
    X_recomp = np.kron(A, B)
    return np.allclose(X, X_recomp)

def cu_decomp(U, cbit=0):
    '''
    decompose the Control-U gate with 2 CNOT gates and some 1-qubit gates
    #FIXME(chuqing): what does CNOT look like when cbit=1?
    '''
    if cbit == 1:
        raise ValueError("unsupported contorl_U gate now")
    gb = GateBuilder()
    tbit = 1 - cbit
    a, b, c, d = get_u3_params(U[2:, 2:])
    A = gb.Rz((d - b) / 2).value
    B = gb.Ry(- c / 2).value @ gb.Rz(- (d + b)/2).value
    C = gb.Rz(b).value @ gb.Ry(c / 2).value
    u_b = get_u3_params(B)
    u_c = get_u3_params(C)
    gates = [ gb.Rz((d - b) / 2, tbit),
              gb.CNOT,
              gb.U3(*u_b[1:], tbit),
              gb.CNOT,
              gb.U3(*u_c[1:], tbit),
              gb.Rz(a, cbit) # rm phase= exp(- aj / 2) here
            ]
    return gates



class GateBuilder():
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
            

    def arr_to_u3(self, U, tbit):
        # get params
        if det(U) == 0:    # cannot be decomposed
            return None
        k = det(U) ** (-0.5)
        U_norm = k * U
        U = U_norm.round(10)
        theta = 2 * atan2(abs(U[1,0]), abs(U[0,0]))
        p_l_sum = 2 * np.angle(U[1, 1])
        p_l_diff = 2 * np.angle(U[1, 0])
        phi = (p_l_sum + p_l_diff) / 2
        lamb = (p_l_sum - p_l_diff) / 2

        # build gate
        return self.U3(theta, phi, lamb, tbit=tbit)
    



