from utils import *

import numpy as np
from enum import Enum
from math import acos, asin, log, atan2
from numpy import exp, sin, cos, sqrt
from scipy.linalg import expm, det

class DeType(Enum):
    two_single: 1


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
            return
        
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
                    pad_len = len(h0) - len(h1)
                    if pad_len > 0:
                        h1 += " " * pad_len
                        q1 += "─" * pad_len
                        t1 += " " * pad_len
                    elif pad_len < 0:
                        h0 += " " * (-1 * pad_len)
                        q0 += "─" * (-1 * pad_len)
                        t0 += " " * (-1 * pad_len)
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
                    # h1 += " " * (len(str) + 4)
                    # q1 += "─" * (len(str) + 4)
                    # t1 += " " * (len(str) + 4)
            else:
                # h0 += " " * (len(str) + 4)
                # q0 += "─" * (len(str) + 4)
                # t0 += " " * (len(str) + 4)
                h1 += "┌" + "─" * (len(str) + 2) + "┐"
                q1 += "┤ " + str + ' ├'
                t1 += "└" + "─" * (len(str) + 2) + "┘"
        pad_len = len(h0) - len(h1)
        if pad_len > 0:
            h1 += " " * pad_len
            q1 += "─" * pad_len
            t1 += " " * pad_len
        elif pad_len < 0:
            h0 += " " * (-1 * pad_len)
            q0 += "─" * (-1 * pad_len)
            t0 += " " * (-1 * pad_len)
            
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
    
    
# Functions for decomposition

def get_u3_params(u, alpha=False):
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
    if alpha:
        return a, b, c, d
    else:
        return b, c, d


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
    a, b, c, d = get_u3_params(U[2:, 2:], alpha=True)
    A = gb.Rz((d - b) / 2).value
    B = gb.Ry(- c / 2).value @ gb.Rz(- (d + b)/2).value
    C = gb.Rz(b).value @ gb.Ry(c / 2).value
    gates = [ gb.Rz((d - b) / 2, tbit),
              gb.CNOT,
              gb.U3(*get_u3_params(B), tbit),
              gb.CNOT,
              gb.U3(*get_u3_params(C), tbit),
              gb.Rz(a, cbit) # rm phase= exp(- aj / 2) here
            ]
    return gates


def general_decomp(U):
    '''
    All notations are according to reference 5
    '''
    gb = GateBuilder()
    I = gb.I.value
    X = gb.X.value
    Y = gb.Y.value
    Z = gb.Z.value
    w = (I - 1j * X) / sqrt(2)

    su4 = su4_convert(U)
    u1, v1, u4_1, v4_1, h1, h2, h3 = su4_decomp(su4)
    
    u2 = 1j / sqrt(2) * (X + Z) @ expm(-1j * (h1 - np.pi / 4) * X)
    u3 = -1j / sqrt(2) * (X + Z)
    v2 = expm(-1j * h3 * Z)
    v3 = expm(1j * h2 * Z)
    u4 = u4_1 @ w
    v4 = v4_1 @ w.conj().T

    gates = [
        gb.U3(*get_u3_params(u1), 0),
        gb.U3(*get_u3_params(v1), 1),
        gb.CNOT,
        gb.U3(*get_u3_params(u2), 0),
        gb.U3(*get_u3_params(v2), 1),
        gb.CNOT,
        gb.U3(*get_u3_params(u3), 0),
        gb.U3(*get_u3_params(v3), 1),
        gb.CNOT,
        gb.U3(*get_u3_params(u4), 0), 
        gb.U3(*get_u3_params(v4), 1)
    ]
    
    return gates


def su4_convert(U):
    A = np.array([[1, 0, 0, 1j],
              [0, 1j, 1, 0],
              [0, 1j, -1, 0],
              [1, 0, 0, -1j]]) / sqrt(2)
    alpha = get_phase(U)
    U_norm = U * np.exp(- 1j * alpha)
    su4 = A.conj().T @ U_norm @ A
    return su4


def kron_decomp_4to2(U: np.ndarray):
    '''
    U = np.kron(g, f1, f2)
    NOTE(chuqing): different from kron_decomp, g is critical and affects accuracy 
                   because SVD decomp is bad in many cases
    :return g, f1, f2
    '''
    # find the largest magnitude 
    a, b = max(((i, j) for i in range(4)
                for j in range(4)), key=lambda t: abs(U[t]))

    # Extract sub-factors touching the reference cell
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = U[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = U[a ^ i, b ^ j]

    # Rescale factors to have unit determinants
    f1 /= np.sqrt(det(f1)) or 1
    f2 /= np.sqrt(det(f2)) or 1

    # Determine global phase
    g = U[a, b] / (f1[a >> 1, b >> 1] * f2[a & 1, b & 1])
    if np.real(g) < 0:
        f1 *= -1
        g = -g

    return g, f1, f2


def simult_svd(A, B):
    """
    Simultaneous SVD of two matrix, based on Eckart-Young theorem.
    A, B => [U D1 V+], [U D2 V+]
    :return: U, V, D1, D2
    """
    d = A.shape[0]
    Ua, Da, Va_h = np.linalg.svd(A)
    
    Ua_h = Ua.conj().T
    Va = Va_h.conj().T
    
    if np.count_nonzero(Da) == d:
        
        G = Ua_h @ B @ Va
        _, P = np.linalg.eig(G)  
        
        U = Ua @ P
        V = Va @ P

        if det(U) < 0:
            U[:, 0] *= -1
        if det(V) < 0:
            V[:, 0] *= -1

        D1 = U.conj().T @ A @ V
        D2 = U.conj().T @ B @ V
        return U, V, D1, D2
    else:
        raise ValueError('A is non-singular matrix')


def su4_decomp(U):
    '''
    U = (A, B) -> (u1, v1)--exp(-iH)--(u4', v4')
    '''
    M = np.array([[1, 0, 0, 1j],
              [0, 1j, 1, 0],
              [0, 1j, -1, 0],
              [1, 0, 0, -1j]]) / sqrt(2)
    M_h = M.conj().T
    A = np.array([[1, 1, -1, 1],
              [1, 1, 1, -1],
              [1, -1, -1, -1],
              [1, -1, 1, 1]])
    
    U, V, D1, D2 = simult_svd(np.real(U), np.imag(U))
    D = D1 + 1j * D2

    _, u4, v4 = kron_decomp_4to2(M @ U @ M_h)
    _, u1, v1 = kron_decomp_4to2(M @ V.T @ M_h)

    _, h1, h2, h3 = - np.linalg.inv(A) @ np.angle(np.diag(D))

    return u1, v1, u4, v4, h1, h2, h3
            
    