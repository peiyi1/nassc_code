from qiskit import QuantumCircuit
import os

path=os.path.dirname(os.path.abspath(__file__))
def circuits():
    current_path=os.path.dirname(os.path.abspath(__file__))
    qc = QuantumCircuit.from_qasm_file(current_path+"/rd84_253.qasm")
    return qc
