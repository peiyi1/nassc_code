# Optimization-Aware Qubit Routing: NASSC
This repository contains the implementation of Optimization-Aware Qubit Routing: NASSC

# Installation
1. Anaconda installation: 

    Anaconda can be downloaded in https://www.anaconda.com/.

2. After installing Anaconda, create an environment and activate the environment:

    $ conda create -y -n env python=3.7
    
    $ conda activate env
    
3. Qiskit installation:

    Clone our source code from github:
    
    $ git clone https://github.com/peiyi1/nassc_code.git
    
    Install the qiskit-terra:
    
    $ cd nassc_code/qiskit-terra
    
    $ pip install cython
    
    $ pip install -r requirements-dev.txt
    
    $ pip install .
    
    Install the qiskit-ibmq-provider:
    
    $ cd ../qiskit-ibmq-provider/
    
    $ pip install -r requirements-dev.txt
    
    $ pip install .
   
4. Install benchmark package and hamap package (the original hamap is from https://github.com/peachnuts/HA):

    $ python setup_benchmark.py develop
    
    $ python setup_hamap.py develop

# Run experiments
Folder test contains scripts to run quantum benchmarks with different routing method. Script generate_raw_data.sh can be used to generate results from all the benchmark.
