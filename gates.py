import numpy as np
from enum import Enum
from utils import *
from math import acos, asin, log, atan2
from numpy import exp, sin, cos, sqrt
from scipy.linalg import expm, det


class DeType(Enum):
    two_single: 1


class Gate():
    def __init__(self, value: np.ndarray, name: str=None, params=None):
        self.__value = value
        self.__name = name
        self.__params = params
        self.__decomps = []
    
    def __str__(self):
        str_to_print = self.name
        if self.params:
            str_to_print += "( "
            for idx in self.params:
                str_to_print += "%.2f " % self.params[idx]
            str_to_print += ")"
        return str_to_print

    @property
    def value(self):
        return self.__value
    
    @property
    def name(self):
        return self.__name

    @property
    def params(self):
        return self.__params

    def column(self, col):
        '''
        return a column as a matrix
        '''
        return self.value[:, col][:, None]

    def is_swap(self):
        swap_m = np.eye(4)[:, [0,2,1,3]]
        return np.allclose(swap_m, self.value)

    def decompose(self, option: str):
        gb = GateBuilder()
        if option == 'two_single':
            matrixs = product_decomp_SVD(self.value)
            gates = [gb.arr_to_u3(m) for m in matrixs]
            if None in gates:
                warning_out("Cannot be decomposed to two 1-qubit gates")
            elif not kron_decomp_check(*matrixs, self.value):
                warning_out("Only approximate decomposition")
            else:
                gates_out(gates)
        else:
            error_out("Unsupported decomposition type")
            return 
        self.__decomps += gates


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
    
    def Rx(self, theta):
        value = expm(-1j * theta / 2 * self.X.value)
        return Gate(value, 'Rx', {'theta': theta})

    def Ry(self, theta):
        value = expm(-1j * theta / 2 * self.Y.value)
        return Gate(value, 'Ry', {'theta': theta})

    def Rz(self, theta):
        value = expm(-1j * theta / 2 * self.Z.value)
        return Gate(value, 'Rz', {'theta': theta})

    def U3(self, theta, phi, lamb):
        value = np.array([[cos(theta / 2), 
                           -exp(1j * lamb) * sin(theta / 2)],
                          [exp(1j * phi) * sin(theta / 2), 
                           exp(1j * (phi + lamb)) * cos(theta / 2)]])
        return Gate(value, 'U3', {'theta': theta,
                                  'phi': phi,
                                  'lambda': lamb})

    def arr_to_u3(self, U):
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
        return self.U3(theta, phi, lamb)