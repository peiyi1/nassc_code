#!/bin/bash

cd test_CouplingMap_FullyConnected

python run_benchmark.py ../yaml_file/bv_n5.yaml
python run_benchmark.py ../yaml_file/3_17_13.yaml
python run_benchmark.py ../yaml_file/decod24-v2_43.yaml
python run_benchmark.py ../yaml_file/mod5d2_64.yaml
python run_benchmark.py ../yaml_file/mod5mils_65.yaml
python run_benchmark.py ../yaml_file/grover_n4.yaml

cd ../test_CouplingMap_montreal

python run_benchmark.py ../yaml_file/bv_n5.yaml
python run_benchmark.py ../yaml_file/3_17_13.yaml
python run_benchmark.py ../yaml_file/decod24-v2_43.yaml
python run_benchmark.py ../yaml_file/mod5d2_64.yaml
python run_benchmark.py ../yaml_file/mod5mils_65.yaml
python run_benchmark.py ../yaml_file/grover_n4.yaml

