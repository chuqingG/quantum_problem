# Quantum Problem

## Problem Statement

Write a program to implement the two-qubit gate decomposition
**Input**: an arbitrary two-qubit gate (an arbitrary 4X4 unitary matrix),  
**Output**: a sequency of single-qubit gates and CNOT gates  
**Example:**  
**Input:**  a two-qubit Control-Z (CZ) gate on two qubits q1 and q2  
**Output:** Hadamard gate (H) on q2, CNOT on q1 q2, Hadamard gate (H) on q2

We designed several sub-tasks you can follow. The main challenge (the 4th sub-task) requires non-trivial efforts and finishing only some of the sub-tasks is accepted:

**1.** (Easy) Write a program to check if the input two-qubit gate can be implemented with **only two single-qubit gates** on the two qubits without using any CNOT gates. If so, find the two single-qubit qubits on the two qubits.

**2.** (Easy) Write a program to check if the input two-qubit gate is a SWAP gate (please find its definition). If so, the output should be 3 CNOT gates without any single-qubit gates. 

**3.** (Medium) Find the definition of a Control-U gate. Read this article http://faculty.cs.tamu.edu/klappi/qalg-f08/controlled-gate.pdf and write a program to detect if the input two-qubit gate is a Control-U gate. If so, decompose the Control-U gate with two CNOT gates and some single-qubit gates

**4.** (Hard) Read [1], following the theorem 1 in [1] (you may also need to read [2] which is referred in [1]), write a program to decompose arbitrary two-qubit gate with three CNOT gates and some single-qubit gates

## Usage
Turn to [example](./examples.ipynb) for usage. If jupyter notebook shows a bad alignment, run [test](test.py) to see the output.

## Reference

1. [A blog about qubit gate in Qiskit](https://blog.csdn.net/qq_36793268/article/details/110352448)
2. [An algorithm of kronecker decomposition](https://www.imageclef.org/system/files/CLEF2016_Kronecker_Decomposition.pdf)
3. Quantum Computation and Quantum Information, Michael A. Nielsen and Issac L. Chuang. 
4. [A website about different qubit gates](https://www.quantum-inspire.com/kbase/rz-gate/)
5. A universal quantum circuit for two-qubit transformations with three CNOT gates, G. Vidal1 and C. M. Dawson.
