#!/usr/bin/env python3
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

# The original file is from https://github.com/1ucian0/rpo.git and has been modified by Peiyi Li

import argparse
import csv
import yaml
from importlib import import_module
from os import path

from benchmark import Result
from qiskit import IBMQ
from qiskit.transpiler import CouplingMap
from hamap import IBMQHardwareArchitecture

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Runs a benchmark.')
parser.add_argument('yamlfile', metavar='file.yaml', nargs=1, help='YAML configuration file')
args = parser.parse_args()
yamlfile = args.yamlfile[0]

with open(yamlfile) as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)

suite = import_module(configuration['suite'])
passmanagers = []
for pm_line in configuration['pass managers']:
    pm_module, pm_func = pm_line.split(':')
    passmanagers.append(getattr(import_module(pm_module), pm_func))

coupling_map = CouplingMap.from_full(25)
from qiskit.test.mock import FakeMontreal
basis_gates = FakeMontreal().configuration().basis_gates

fields= []
fields.append( configuration['fields'][1])
fields.append( configuration['fields'][2])
fields.append( configuration['fields'][4])

times = configuration.get('times', 1)
resultfile = path.join('results', '%s.csv' % path.basename(yamlfile).split('.')[0])

print('suite:', configuration['suite'])
print('times:', str(times))
print('result file:', resultfile)

with open(resultfile, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    circuit = suite.circuits()
    result = Result(circuit, basis_gates, coupling_map, routing_method="sabre")
    result.run_pms(passmanagers, times=times)
    writer.writerow(result.row(fields))
