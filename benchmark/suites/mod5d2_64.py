from qiskit import QuantumCircuit
import os

def circuits():
    current_path=os.path.dirname(os.path.abspath(__file__))
    qc = QuantumCircuit.from_qasm_file(current_path+"/mod5d2_64.qasm")
    return qc
