import numpy as np
import random
from math import sqrt, acos, exp, asin, log, sin
from scipy.linalg import expm, det
from scipy.stats import unitary_group
from itertools import permutations

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

