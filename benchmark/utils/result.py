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

from qiskit.transpiler import PassManagerConfig
from qiskit.transpiler.coupling import CouplingMap
from qiskit.providers.aer.noise import NoiseModel
from qiskit import execute, Aer

class Result:
    def __init__(self, circuit, basis_gates, coupling_map, routing_method, shots=1, noise_model=None, hardware=None ):
        self.input_circuit = circuit
        self.n_qubits = len(circuit.qubits)
        self.depth = circuit.depth()
        self.pms_results = {}
        self.basis_gates = basis_gates
        self.coupling_map = coupling_map
        self.routing_method=routing_method
        self.shots=shots
        self.noise_model=noise_model
        self.hardware = hardware
    @staticmethod
    def pm_config(basis_gates,coupling_map,routing_method,hardware):
        return PassManagerConfig(
            initial_layout=None,
            basis_gates= basis_gates,
            coupling_map= coupling_map,
            routing_method=routing_method,
            enable_factor_block=True,
            enable_factor_commute_0=True,
            enable_factor_commute_1=True,
            factor_block=1,
            factor_commute_0=1,
            factor_commute_1=1,
            seed_transpiler=11,
            hardware = hardware
        )

    def run_pm_with_time(self, passmanager):
        times = {'total': 0}
        repetition = {}

        def collect_time(**kwargs):
            times['total'] += kwargs['time']
            passname = type(kwargs['pass_']).__name__
            if passname in times:
                times[passname] += kwargs['time']
                repetition[passname] += 1
            else:
                times[passname] = kwargs['time']
                repetition[passname] = 0

        pm = passmanager(Result.pm_config(self.basis_gates,self.coupling_map,self.routing_method, self.hardware))
        transpiled = None
        try:
            transpiled = pm.run(self.input_circuit, callback=collect_time)
        except:
            pass

        return transpiled, times, repetition

    def run_pms(self, passmanagers, times=10):
        for pm in passmanagers:
            result = {'transpiled': [], 'times': {}, 'repetitions': {}}
            for times in range(times):
                transpiled, calls, repetitions = self.run_pm_with_time(pm)
                if transpiled is not None:
                    result['transpiled'].append(transpiled)
                    for passname, time in calls.items():
                        if passname in result['times']:
                            result['times'][passname].append(time)
                        else:
                            result['times'][passname] = [time]
                    for passname, rep in repetitions.items():
                        if passname in result['repetitions']:
                            result['repetitions'][passname].append(rep)
                        else:
                            result['repetitions'][passname] = [rep]

            self.pms_results[pm.__name__] = result

    def row(self, fields):
        return {field: getattr(self, field) for field in fields}

    @property
    def level3_cxs(self):
        cx_results = []
        for cx_result in self.pms_results['level_3_pass_manager']['transpiled']:
            try:
                cx_count = cx_result.count_ops()['cx']
            except:
                cx_count = 0
            cx_results.append(cx_count)
        return cx_results

    @property
    def level3_depth(self):
        depth_results = []
        for sample in self.pms_results['level_3_pass_manager']['transpiled']:
            depth_results.append(sample.depth())
        return depth_results[0] if len(depth_results) == 1 else depth_results

    @property
    def level3_time(self):
        return self.pms_results['level_3_pass_manager']['times'].get('total', None)

    def correct_key(self):
        basic_circ = self.pms_results['level_3_pass_manager']['transpiled'][0]
        result = execute(basic_circ, Aer.get_backend('qasm_simulator'),shots=self.shots ).result().get_counts()
        for key in result:
            if result[key]/self.shots > 0.9:
                return key
    @property
    def successful_rate(self):
        successful_rate_result=[]
        for basic_circ in self.pms_results['level_3_pass_manager']['transpiled']:
            result = execute(basic_circ, Aer.get_backend('qasm_simulator'),
                             basis_gates=self.noise_model.basis_gates,
                             coupling_map=self.coupling_map,
                             noise_model=self.noise_model,
                             shots=self.shots  ).result().get_counts()
            key = self.correct_key()
            successful_rate = result[key]/self.shots
            successful_rate_result.append(successful_rate)
        return successful_rate_result
