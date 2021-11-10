from qiskit import QuantumCircuit
import os

path=os.path.dirname(os.path.abspath(__file__))
def circuits():
    current_path=os.path.dirname(os.path.abspath(__file__))
    qc = QuantumCircuit.from_qasm_file(current_path+"/vqe_n8.qasm")
    return qc
