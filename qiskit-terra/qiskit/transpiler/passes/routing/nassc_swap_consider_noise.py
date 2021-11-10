# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# notice: the original code is from Qiskit and has been modified by Peiyi Li

import logging
from collections import defaultdict
from copy import copy, deepcopy
import numpy as np

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode, DAGCircuit #pli11: add DAGCircuit 
from qiskit.quantum_info.operators import Operator #pli11
from qiskit.circuit import Gate, QuantumRegister, QuantumCircuit #pli11
from qiskit.extensions import UnitaryGate #pli11
from qiskit.converters import circuit_to_dag #pli11

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = 20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


class NASSCSwapConsiderNoise(TransformationPass):
    
    def __init__(self, coupling_map, heuristic="basic", seed=None, fake_run=False, enable_factor_block=False, enable_factor_commute_0=False, enable_factor_commute_1=False, factor_block=0, factor_commute_0=0, factor_commute_1=0, decomposer2q=None, approximation_degree=None, hardware = None):
    
        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()

        self.heuristic = heuristic
        self.seed = seed
        self.fake_run = fake_run
        self.applied_predecessors = None
        self.qubits_decay = None
        self._bit_indices = None
    
        #pli11: used for commutation analysis
        self.cache = {} 
        self.commutation_set = None
        #pli11: enable the recalculation of the swap score
        self.enable_factor_block = enable_factor_block
        self.enable_factor_commute_0 = enable_factor_commute_0
        self.enable_factor_commute_1 = enable_factor_commute_1
        #pli11: used for recalculating the swap score
        self.factor_block = factor_block
        self.factor_commute_0 = factor_commute_0
        self.factor_commute_1 = factor_commute_1
        #pli11: for collect 2q blocks
        self.pending_1q = None
        self.block_id = None
        self.current_id = None
        self.block_list = None
        #pli11: for synthesis 2q block
        self.decomposer2q = decomposer2q
        self.approximation_degree = approximation_degree
        self.norm_swap_number =None
        self.norm_error_cost =None
        self.hardware = hardware #pli11: use device montreal to test
        
    def run(self, dag):
        
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("NASSC swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        rng = np.random.default_rng(self.seed)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None
        if not self.fake_run:
            mapped_dag = dag._copy_circuit_metadata()

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)

        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        self.qubits_decay = {qubit: 1 for qubit in dag.qubits}

        #pli11: initiate the commutation set
        self.commutation_set = defaultdict(list)
        for wire in dag.wires:
            self.commutation_set[wire] = []
        # Add edges to the dictionary for each qubit
        for node in dag.topological_op_nodes():
            for (_, _, edge_wire) in dag.edges(node):
                self.commutation_set[(node, edge_wire)] = -1
        #pli11: init self.pending_1q, self.block_id, self.current_id, self.block_list
        self.pending_1q = [list() for _ in range(dag.num_qubits())]
        self.block_id = [-(i + 1) for i in range(dag.num_qubits())]
        self.current_id = 0
        self.block_list = list()
        #init value of norm
        self.norm_swap_number = self.coupling_map.obtain_norm_swap_number()
        self.norm_error_cost =self.coupling_map.obtain_norm_error_cost()
        # Start algorithm from the front layer and iterate until all gates done.
        num_search_steps = 0
        front_layer = dag.front_layer()
        self.applied_predecessors = defaultdict(int)
        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1
        while front_layer:
            execute_gate_list = []

            # Remove as many immediately applicable gates as possible
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    if self.coupling_map.graph.has_edge(current_layout[v0], current_layout[v1]):
                        execute_gate_list.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)

            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(mapped_dag, node, current_layout, canonical_register)
                    front_layer.remove(node)
                    for successor in self._successors(node, dag):
                        self.applied_predecessors[successor] += 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)

                    if node.qargs:
                        self._reset_qubits_decay()

                

                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            extended_set = self._obtain_extended_set(dag, front_layer)
            swap_candidates = self._obtain_swaps(front_layer, current_layout)
            swap_scores = dict.fromkeys(swap_candidates, 0)
            predecessor_block=list() #pli11: used for collecting 2q block
            swap_commutation = dict.fromkeys(swap_candidates, None) #pli11: used for commutation pattern check
            for swap_qubits in swap_scores:
                trial_layout = current_layout.copy()
                #pli11: record the score in current layout
                score_current_layout = self._score_heuristic(
                    "basic", front_layer, extended_set, trial_layout, swap_qubits
                )
                trial_layout.swap(*swap_qubits)
                #pli11: compute the score in basic config
                score = self._score_heuristic(
                    "basic", front_layer, extended_set, trial_layout, swap_qubits
                )
                #pli11: if this swap inserted in opposite direction, do not consider 2q block and commutation opt.
                if score - score_current_layout > 0:
                    swap_scores[swap_qubits] = self._score_heuristic_recalculate(
                        swap_scores[swap_qubits], self.heuristic, front_layer, extended_set, trial_layout, swap_qubits
                    )
                    continue
                else:
                    swap_scores[swap_qubits] = self._score_heuristic_consider_noise(
                        self.heuristic, front_layer, extended_set, trial_layout, swap_qubits
                    )
                #pli11
                swap_node = DAGOpNode(op=SwapGate(), qargs=swap_qubits)
                swap_node = _transform_gate_for_layout(swap_node, current_layout, canonical_register)
                #pli11: see if there is 2q block that can be merged, and store the info into predecessor_block
                predecessor_block.clear()
                swap_commutation[swap_qubits] = (None, None, None, None, None, None, None)
                qids = [self._bit_indices[q] for q in swap_node.qargs]
                if ((self.enable_factor_block)
                    and (self.block_id[qids[0]] == self.block_id[qids[1]])
                   ):
                    predecessor_block.extend(self.block_list[self.block_id[qids[0]]])
                
                #pli11:reduce the swap score based on 2q block or comuation pattern 0 and 1
                if len(predecessor_block)>0:
                    if not (predecessor_block[-1].name == "swap"):
                        gate_2q_num_before_adding_swap=0
                        gate_2q_num_after_adding_swap=0
                        
                        v = QuantumRegister(2, "v")
                        subcirc = QuantumCircuit(v)
                        to_vid = dict()
                        for i, qubit in enumerate(swap_node.qargs):
                            to_vid[qubit] = i
                        
                        for gate in predecessor_block:
                            vids = [to_vid[q] for q in gate.qargs]
                            subcirc.append(gate.op, vids)
                            if len(gate.qargs) == 2:
                                gate_2q_num_before_adding_swap += 1
                                
                        subcirc.append(SwapGate(), [v[0], v[1]])
                        unitary = UnitaryGate(Operator(subcirc))  
                        
                        basis_fidelity = self.approximation_degree
                        su4_mat = unitary.to_matrix()
                        synth_circ = self.decomposer2q(su4_mat, basis_fidelity=basis_fidelity)
                        
                        synth_dag = circuit_to_dag(synth_circ)
                        for node in synth_dag.op_nodes():
                            if len(node.qargs) ==2:
                                gate_2q_num_after_adding_swap+=1
                                
                        decreased_score= 3 - (gate_2q_num_after_adding_swap-gate_2q_num_before_adding_swap)
                        #pli11: 0.5 corresponding to ha parameter
                        swap_scores[swap_qubits] -= decreased_score*self.factor_block/3/self.norm_swap_number*0.5 
                        cnot_fidelity=1 - self.hardware.get_link_error_rate(trial_layout[swap_qubits[0]], trial_layout[swap_qubits[1]])
                        error_cost_before=1-pow(cnot_fidelity,gate_2q_num_before_adding_swap+3)
                        error_cost_after=1-pow(cnot_fidelity,gate_2q_num_after_adding_swap)
                        swap_scores[swap_qubits] -= (error_cost_before-error_cost_after)/self.norm_error_cost*0.5
                        
                else:  
                    if(self.enable_factor_commute_0 or self.enable_factor_commute_1):
                        #pli11: swap_check: commutation pattern 0 and pattern 1 checking
                        swap_commutation[swap_qubits]=self._swap_check(swap_node)
                        #pli11: recalculating the swap score
                        if (self.enable_factor_commute_0 and swap_commutation[swap_qubits][0]):
                            swap_scores[swap_qubits] -= 2*self.factor_commute_0/3/self.norm_swap_number*0.5 
                            cnot_fidelity=1 - self.hardware.get_link_error_rate(trial_layout[swap_qubits[0]], trial_layout[swap_qubits[1]])
                            error_cost_before=1-pow(cnot_fidelity,3+3)
                            error_cost_after=1-pow(cnot_fidelity,4)
                            swap_scores[swap_qubits] -= (error_cost_before-error_cost_after)/self.norm_error_cost*0.5
                        
                        elif (self.enable_factor_commute_1 and swap_commutation[swap_qubits][1]):
                            swap_scores[swap_qubits] -= 2*self.factor_commute_1/3/self.norm_swap_number*0.5
                            cnot_fidelity=1 - self.hardware.get_link_error_rate(trial_layout[swap_qubits[0]], trial_layout[swap_qubits[1]])
                            error_cost_before=1-pow(cnot_fidelity,3+1)
                            error_cost_after=1-pow(cnot_fidelity,2)
                            swap_scores[swap_qubits] -= (error_cost_before-error_cost_after)/self.norm_error_cost*0.5
                        
                swap_scores[swap_qubits] = self._score_heuristic_recalculate_consider_noise(
                    swap_scores[swap_qubits], self.heuristic, front_layer, extended_set, trial_layout, swap_qubits
                )    
            min_score = min(swap_scores.values())
            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
            best_swap = rng.choice(best_swaps)
            swap_node = DAGOpNode(op=SwapGate(), qargs=best_swap)
            
            if((self.enable_factor_commute_0 and swap_commutation[tuple(best_swap)][0])
               or (self.enable_factor_commute_1 and swap_commutation[tuple(best_swap)][1])
              ):
            
                #pli11: consider the commutation rule when applying gate
                self._apply_gate_consider_commutation(swap_commutation[tuple(best_swap)],
                                                      mapped_dag, swap_node, current_layout, canonical_register)
            else:
                self._apply_gate(mapped_dag, swap_node, current_layout, canonical_register)
            
            current_layout.swap(*best_swap)

            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL == 0:
                self._reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE

            
        self.property_set["final_layout"] = current_layout
           
        if not self.fake_run:
            return mapped_dag
        return dag
  
    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        if self.fake_run:
            return
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        new_node = mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
        
        #pli11: collect 2q blocks
        qids = [self._bit_indices[q] for q in new_node.qargs]
        if (
                not isinstance(new_node.op, Gate)
                or len(qids) > 2
                or new_node.op.condition
                or new_node.op.is_parameterized()
            ):
            for qid in qids:
                if self.block_id[qid] > 0:
                    self.block_list[self.block_id[qid]].extend(self.pending_1q[qid])
                self.block_id[qid] = -(qid + 1)
                self.pending_1q[qid].clear()
                
        elif len(qids) == 1:
            b_id = self.block_id[qids[0]]
            if b_id < 0:
                self.pending_1q[qids[0]].append(new_node)
            else:
                self.block_list[b_id].append(new_node)
        elif len(qids) == 2:
            if self.block_id[qids[0]] == self.block_id[qids[1]]:
                self.block_list[self.block_id[qids[0]]].append(new_node)
            else:
                self.block_id[qids[0]] = self.current_id
                self.block_id[qids[1]] = self.current_id
                new_block = list()
                if self.pending_1q[qids[0]]:
                    new_block.extend(self.pending_1q[qids[0]])
                    self.pending_1q[qids[0]].clear()
                if self.pending_1q[qids[1]]:
                    new_block.extend(self.pending_1q[qids[1]])
                    self.pending_1q[qids[1]].clear()
                new_block.append(new_node)
                self.block_list.append(new_block)
                self.current_id += 1
        
        #pli11
        # Construct the commutation set
        for wire in new_node.qargs:
            current_comm_set = self.commutation_set[wire]
            if not current_comm_set:
                current_comm_set.append([new_node])
                
            if new_node not in current_comm_set[-1]:
                prev_gate = current_comm_set[-1][-1]
                does_commute = False
                try:
                    does_commute = _commute(new_node, prev_gate, self.cache)
                except TranspilerError:
                    pass
                if does_commute:
                    current_comm_set[-1].append(new_node)

                else:
                    current_comm_set.append([new_node])
        
            temp_len = len(current_comm_set)
            self.commutation_set[(new_node, wire)] = temp_len - 1
     
    #pli11: apply gate consider the commutation rule
  
    def _apply_gate_consider_commutation(self,swap_commutation, mapped_dag, node, current_layout, canonical_register):
        if self.fake_run:
            return
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        
        #for saving the single qubit gates
        new_single_gates_0 = list()
        new_single_gates_1 = list()
        
        if swap_commutation:
            #deal with the first swap in commutation pattern 0 
            if swap_commutation[3]:
                swap_circuit = DAGCircuit()
                v = QuantumRegister(2, "v")
                swap_circuit.add_qreg(v)
                swap_circuit.apply_operation_back(SwapGate(), [v[1], v[0]], [])
                
                if not swap_commutation[2].cargs:
                    swap_commutation[2].cargs = list()
                
                mapped_dag.substitute_node_with_dag(swap_commutation[2], swap_circuit, wires=[v[0], v[1]])
            
            #deal with the inserted swap:
            if swap_commutation[4]:
                new_node = mapped_dag.apply_operation_back(new_node.op, [new_node.qargs[1],new_node.qargs[0]], new_node.cargs, True)
                
                #remove the single qubit gates:
                #and add the single qubit gates and save the added gates:
                for nd in swap_commutation[5][::-1]:
                    mapped_dag.remove_op_node(nd)
                    new_single_gate = mapped_dag.apply_operation_back(nd.op,[new_node.qargs[0]])
                    new_single_gates_0.append(new_single_gate)
                for nd in swap_commutation[6][::-1]:
                    mapped_dag.remove_op_node(nd)
                    new_single_gate = mapped_dag.apply_operation_back(nd.op,[new_node.qargs[1]])
                    new_single_gates_1.append(new_single_gate)
                    
            else:
                new_node = mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs, True)
            
                #remove the single qubit gates:
                #and add the single qubit gates and save the added gates:
                for nd in swap_commutation[5][::-1]:
                    mapped_dag.remove_op_node(nd)
                    new_single_gate = mapped_dag.apply_operation_back(nd.op,[new_node.qargs[1]])
                    new_single_gates_1.append(new_single_gate)
                for nd in swap_commutation[6][::-1]:
                    mapped_dag.remove_op_node(nd)
                    new_single_gate = mapped_dag.apply_operation_back(nd.op,[new_node.qargs[0]])
                    new_single_gates_0.append(new_single_gate)
             
        else:
            new_node = mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs, True)
            
        # Construct the commutation set
        for wire in new_node.qargs:
            current_comm_set = self.commutation_set[wire]
            if not current_comm_set:
                current_comm_set.append([new_node])
                
            if new_node not in current_comm_set[-1]:
                prev_gate = current_comm_set[-1][-1]
                does_commute = False
                try:
                    does_commute = _commute(new_node, prev_gate, self.cache)
                except TranspilerError:
                    pass
                if does_commute:
                    current_comm_set[-1].append(new_node)

                else:
                    current_comm_set.append([new_node])
        
            temp_len = len(current_comm_set)
            self.commutation_set[(new_node, wire)] = temp_len - 1
        
        # Construct the commutation set with single qubit gates
        for new_single_gate in new_single_gates_0:
            current_comm_set = self.commutation_set[new_node.qargs[0]]
            current_comm_set.append([new_single_gate])
            temp_len = len(current_comm_set)
            self.commutation_set[(new_single_gate, new_node.qargs[0])] = temp_len - 1
        for new_single_gate in new_single_gates_1:
            current_comm_set = self.commutation_set[new_node.qargs[1]]
            current_comm_set.append([new_single_gate])
            temp_len = len(current_comm_set)
            self.commutation_set[(new_single_gate, new_node.qargs[1])] = temp_len - 1
        
        #pli11: collect 2q blocks
        qids = [self._bit_indices[q] for q in new_node.qargs]
        if (
                not isinstance(new_node.op, Gate)
                or len(qids) > 2
                or new_node.op.condition
                or new_node.op.is_parameterized()
            ):
            for qid in qids:
                self.block_id[qid] = -(qid + 1)
        elif len(qids) == 2:
            if self.block_id[qids[0]] == self.block_id[qids[1]]:
                self.block_list[self.block_id[qids[0]]].append(new_node)
            else:
                self.block_id[qids[0]] = self.current_id
                self.block_id[qids[1]] = self.current_id
                new_block = list()
                new_block.append(new_node)
                self.block_list.append(new_block)
                self.current_id += 1
                
    def _reset_qubits_decay(self):
        """Reset all qubit decay factors to 1 upon request (to forget about
        past penalizations).
        """
        self.qubits_decay = {k: 1 for k in self.qubits_decay.keys()}

    def _successors(self, node, dag):
        for _, successor, edge_data in dag.edges(node):
            if not isinstance(successor, DAGOpNode):
                continue
            if isinstance(edge_data, Qubit):
                yield successor

    def _is_resolved(self, node):
        """Return True if all of a node's predecessors in dag are applied."""
        return self.applied_predecessors[node] == len(node.qargs)

    def _obtain_extended_set(self, dag, front_layer):
        """Populate extended_set by looking ahead a fixed number of gates.
        For each existing element add a successor until reaching limit.
        """
        extended_set = list()
        incremented = list()
        tmp_front_layer = front_layer
        done = False
        while tmp_front_layer and not done:
            new_tmp_front_layer = list()
            for node in tmp_front_layer:
                for successor in self._successors(node, dag):
                    incremented.append(successor)
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        new_tmp_front_layer.append(successor)
                        if len(successor.qargs) == 2:
                            extended_set.append(successor)
                if len(extended_set) >= EXTENDED_SET_SIZE:
                    done = True
                    break
            tmp_front_layer = new_tmp_front_layer
        for node in incremented:
            self.applied_predecessors[node] -= 1
        return extended_set

    def _obtain_swaps(self, front_layer, current_layout):
        """Return a set of candidate swaps that affect qubits in front_layer.

        For each virtual qubit in front_layer, find its current location
        on hardware and the physical qubits in that neighborhood. Every SWAP
        on virtual qubits that corresponds to one of those physical couplings
        is a candidate SWAP.

        Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
        """
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted([virtual, virtual_neighbor], key=lambda q: self._bit_indices[q])
                    candidate_swaps.add(tuple(swap))

        return candidate_swaps

    def _compute_cost(self, layer, layout):
        cost = 0
        for node in layer:
            cost += self.coupling_map.distance(layout[node.qargs[0]], layout[node.qargs[1]])
        #pli11: change the return cost to 3*cost in order to recalculate the score
        return 3*cost

    def _score_heuristic(self, heuristic, front_layer, extended_set, layout, swap_qubits=None):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        first_cost = self._compute_cost(front_layer, layout)
        if heuristic == "basic":
            return first_cost

        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return (
                max(self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]])
                * total_cost
            )

        raise TranspilerError("Heuristic %s not recognized." % heuristic)
        
    #pli11: def _score_heuristic_recalculate: recalculating the score of swap 
    def _score_heuristic_recalculate(self, score, heuristic, front_layer, extended_set, layout, swap_qubits=None):
        """Return a heuristic score for a trial layout.
        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        first_cost = score
        if heuristic == "basic":
            return first_cost

        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return (
                max(self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]])
                * total_cost
            )

        raise TranspilerError("Heuristic %s not recognized." % heuristic)
        
    #pli11: coupute distance matrix consider noise
    def _compute_cost_consider_noise(self, layer, layout):
        cost = 0
        for node in layer:
            cost += self.coupling_map.distance_consider_noise(layout[node.qargs[0]], layout[node.qargs[1]],self.hardware)
        return cost

    def _score_heuristic_consider_noise(self, heuristic, front_layer, extended_set, layout, swap_qubits=None):
        """Return a heuristic score for a trial layout.
        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        first_cost = self._compute_cost_consider_noise(front_layer, layout)
        if heuristic == "basic":
            return first_cost

        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost_consider_noise(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return (
                max(self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]])
                * total_cost
            )

        raise TranspilerError("Heuristic %s not recognized." % heuristic)
    
    def _score_heuristic_recalculate_consider_noise(self, score, heuristic, front_layer, extended_set, layout, swap_qubits=None):
        """Return a heuristic score for a trial layout.
        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        first_cost = score
        if heuristic == "basic":
            return first_cost

        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost_consider_noise(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return (
                max(self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]])
                * total_cost
            )

        raise TranspilerError("Heuristic %s not recognized." % heuristic)
          
    #pli11: add _swap_check: commutation pattern checking 
  
    def _swap_check(self, node):
        
        flag_swap_check_0 = False # commutation pattern 0 checking
        flag_swap_check_1 = False # commutation pattern 1 checking
        #flag_swap_check_2qblock = False # 2q block checking
        
        #flag_swap_check_2qblock_has_swap= False #check if the 2q block has swap gate, and this swap is not a swap inserted in routing step
        
        swap_in_pattern0= None
        swap_in_pattern0_decomposition_method = False
        swap_decomposition_method= False 
        
        single_gate_list_0=list()
        single_gate_list_1=list()
        
        wire_commutation_set_0 = self.commutation_set[node.qargs[0]]
        wire_commutation_set_1 = self.commutation_set[node.qargs[1]]
        
        #check if wire0 and wire 1 have commutation set
        if (wire_commutation_set_0 and wire_commutation_set_1):
            #check the first wire of the swap
            end_search_wire0=False
            current_index_0 = len(wire_commutation_set_0) - 1
            gate_2q_wire0 = None
            while not end_search_wire0:
                current_list_node = wire_commutation_set_0[current_index_0]
                for nd in reversed(current_list_node):
                    if (isinstance(nd.op, Gate)
                        and len(nd.qargs) ==1
                       ):
                        single_gate_list_0.append(nd)
                    else:
                        end_search_wire0=True
                        gate_2q_wire0 = nd
                        break
                if not end_search_wire0:
                    current_index_0 -= 1 
                    if current_index_0 < 0:
                        current_index_0 += 1 
                        end_search_wire0=True
                  
            #check the second wire of the swap
            end_search_wire1=False
            current_index_1 = len(wire_commutation_set_1) - 1
            gate_2q_wire1 = None
            while not end_search_wire1:
                current_list_node = wire_commutation_set_1[current_index_1]
                for nd in reversed(current_list_node):
                    if (isinstance(nd.op, Gate)
                        and len(nd.qargs) ==1
                       ):
                        single_gate_list_1.append(nd)
                    else:
                        end_search_wire1=True
                        gate_2q_wire1 = nd
                        break
                if not end_search_wire1:
                    current_index_1 -= 1 
                    if current_index_1 < 0:
                        current_index_1 += 1 
                        end_search_wire1=True
            
            #check the common node of the two wire
            common_node=list(set(wire_commutation_set_0[current_index_0]) & set(wire_commutation_set_1[current_index_1]) )
            
            if not common_node:
                
                if ((gate_2q_wire0) 
                     and (gate_2q_wire0.name == "swap")
                     and (set(gate_2q_wire0.qargs) == set(node.qargs))
                   ):
                    for n in wire_commutation_set_1[current_index_1]:
                        if ((n.name == "cx")
                            and (self.commutation_set[(n,node.qargs[1])]-
                                 self.commutation_set[(gate_2q_wire0,node.qargs[1])]==1)
                           ):
                            flag_swap_check_0=True
                            swap_in_pattern0 = gate_2q_wire0
                            swap_in_pattern0_decomposition_method=self._swap_decomposition_method(gate_2q_wire0, n)
                            swap_decomposition_method = self._swap_decomposition_method(node, n)
                            break
                else:
                    if ((gate_2q_wire1) 
                         and (gate_2q_wire1.name == "swap")
                         and (set(gate_2q_wire1.qargs) == set(node.qargs))
                       ):
                        for n in wire_commutation_set_0[current_index_0]:
                            if (n.name == "cx"
                                and (self.commutation_set[(n,node.qargs[0])]-
                                 self.commutation_set[(gate_2q_wire1,node.qargs[0])]==1)
                               ):
                                flag_swap_check_0=True
                                swap_in_pattern0 = gate_2q_wire1
                                swap_in_pattern0_decomposition_method=self._swap_decomposition_method(gate_2q_wire1, n)
                                swap_decomposition_method = self._swap_decomposition_method(node, n)
                                break    
            else:  
                for n in common_node:
                    if ((n.name == "cx")
                        and (set(n.qargs) == set(node.qargs))
                        ):
                        flag_swap_check_1=True
                        swap_decomposition_method = self._swap_decomposition_method(node, n)
                        break
                        
        if( (not flag_swap_check_0)
           and (not flag_swap_check_1)
           #and (not flag_swap_check_2qblock)
          ):
            single_gate_list_0.clear()
            single_gate_list_1.clear()
            
        return (flag_swap_check_0, flag_swap_check_1, swap_in_pattern0, swap_in_pattern0_decomposition_method, swap_decomposition_method, single_gate_list_0, single_gate_list_1) #, flag_swap_check_2qblock, flag_swap_check_2qblock_has_swap)
    
    #pli11: add _swap_decomposition_method to determine how to decompose swap
    def _swap_decomposition_method(self, swap_node, another_node):
        swap_decomposition_method = False
        
        if(another_node.qargs[0]==swap_node.qargs[1]
           or another_node.qargs[1]==swap_node.qargs[0]
          ):
            swap_decomposition_method = True
            
        return swap_decomposition_method    
        
def _transform_gate_for_layout(op_node, layout, device_qreg):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = copy(op_node)

    premap_qargs = op_node.qargs
    mapped_qargs = map(lambda x: device_qreg[layout[x]], premap_qargs)
    mapped_op_node.qargs = list(mapped_qargs)

    return mapped_op_node
#pli11: add method _commute
def _commute(node1, node2, cache):

    if not isinstance(node1, DAGOpNode) or not isinstance(node2, DAGOpNode):
        return False

    for nd in [node1, node2]:
        if nd.op._directive or nd.name in {"measure", "reset", "delay"}:
            return False

    if node1.op.condition or node2.op.condition:
        return False

    if node1.op.is_parameterized() or node2.op.is_parameterized():
        return False

    qarg = list(set(node1.qargs + node2.qargs))
    qbit_num = len(qarg)

    qarg1 = [qarg.index(q) for q in node1.qargs]
    qarg2 = [qarg.index(q) for q in node2.qargs]

    id_op = Operator(np.eye(2 ** qbit_num))

    node1_key = (node1.op.name, str(node1.op.params), str(qarg1))
    node2_key = (node2.op.name, str(node2.op.params), str(qarg2))
    if (node1_key, node2_key) in cache:
        op12 = cache[(node1_key, node2_key)]
    else:
        op12 = id_op.compose(node1.op, qargs=qarg1).compose(node2.op, qargs=qarg2)
        cache[(node1_key, node2_key)] = op12
    if (node2_key, node1_key) in cache:
        op21 = cache[(node2_key, node1_key)]
    else:
        op21 = id_op.compose(node2.op, qargs=qarg2).compose(node1.op, qargs=qarg1)
        cache[(node2_key, node1_key)] = op21

    if_commute = op12 == op21

    return if_commute
