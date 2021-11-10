# -*- coding: utf-8 -*-
  
# (C) Copyright Ji Liu and Luciano Bello 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# the original file is from https://github.com/1ucian0/rpo.git and has been modified by Peiyi Li

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

import numpy as np

def grover_diffusion_operator(circuit, qreg, nbits):
    for i in range(0, nbits):
        circuit.h(i)
        circuit.x(i)
    circuit.h(qreg[nbits - 1])
    circuit.mct(qreg[0: nbits - 1], qreg[nbits - 1], None, 'noancilla')
    circuit.h(qreg[nbits - 1])
    for i in range(0, nbits):
        circuit.x(i)
        circuit.h(i)
    return circuit


def grover_oracle(circuit, qreg, nbits, hidden_value):
    for i in range(0, nbits):
        circuit.h(i)
    control_list = []
    for j in range(0, nbits):
        if (hidden_value & (1 << j)):
            control_list.append(qreg[j])
    circuit.h(control_list[-1])
    circuit.mct(control_list[0:-1], control_list[-1], None, 'noancilla')
    circuit.h(control_list[-1])
    for i in range(0, nbits):
        circuit.h(i)
    return circuit


def grover(nbits=6, expected_output=None, measure=True):
    """
        This is a nbit Grover's algorithm that find (with probability close to 1) a specific item
        within a randomly ordered database of N items using O(√N) operations
        reference: https://ieeexplore.ieee.org/abstract/document/8622457
    """
    if expected_output is None:
        expected_output = 2 ** nbits - 1
    qr = QuantumRegister(nbits, 'qr')
    circuit = QuantumCircuit(qr)
    grover_oracle(circuit, qr, nbits, expected_output)
    #iteration = int(round(np.pi/4 * math.sqrt(2 ** nbits)))
    #for i in range(0, iteration):
    grover_diffusion_operator(circuit, qr, nbits)
    if measure is True:
        circuit.measure_all()
    return circuit


def circuits():
    return grover(6)
