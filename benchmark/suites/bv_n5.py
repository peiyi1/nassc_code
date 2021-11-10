import qiskit
from qiskit import QuantumCircuit
import math
import numpy as np
    
def circuits():
    n = 5
    s = '11111'

    bv_circuit = QuantumCircuit(n+1, n)
    bv_circuit.h(n)
    bv_circuit.z(n)

    for i in range(n):
        bv_circuit.h(i)
    bv_circuit.barrier()

    s = s[::-1]
    for q in range(n):
        if s[q] == '0':
            bv_circuit.i(q)
        else:
            bv_circuit.cx(q, n)
    
    bv_circuit.barrier()

    for i in range(n):
        bv_circuit.h(i)

    for i in range(n):
        bv_circuit.measure(i, i)
    
    return bv_circuit
