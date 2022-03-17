import numpy as np
from math import sqrt, acos, exp, asin, log, sin, atan2
from scipy.linalg import expm, det
from itertools import permutations
# from gates import GateBuilder


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