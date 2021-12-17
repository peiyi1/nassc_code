[![DOI](https://zenodo.org/badge/426498854.svg)](https://zenodo.org/badge/latestdoi/426498854)
# Optimization-Aware Qubit Routing: NASSC
This repository contains the source code, benchmarks and scripts to reproduce experiments from the HPCA 2022 paper "Not All SWAPs Have the Same Cost: A Case for Optimization-Aware Qubit Routing" by Ji Liu, Peiyi Li, and Huiyang Zhou.

# Installation
1. Anaconda installation: 

    Anaconda can be downloaded in https://www.anaconda.com/

2. After installing Anaconda, create an environment and activate the environment:

    $ conda create -y -n env python=3.7
    
    $ conda activate env
    
3. Qiskit installation:
    
    After downloading our repository from Zenodo, go into the downloaded repository and install qiskit-terra and qiskit-ibmq-provider:

    (1). Go to the folder /qiskit-terra/ and install the qiskit-terra:
    
        $ pip install cython
    
        $ pip install -r requirements-dev.txt
    
        $ pip install .
    
    (2). Go to the folder /qiskit-ibmq-provider/ and install the qiskit-ibmq-provider:
    
        $ pip install -r requirements-dev.txt
    
        $ pip install .
   
4. Package installation:
    
    (1). Install benchmark package:

        $ python setup_benchmark.py develop
        
    (2). Install hamap package(the original hamap is from https://github.com/peachnuts/HA):
    
        $ python setup_hamap.py develop

# Experiments workflow
The appendix of the HPCA 2022 paper "Not All SWAPs Have the Same Cost: A Case for Optimization-Aware Qubit Routing" contains an experiments workflow, please refer to the experiments workflow to reproduce the experiments results.
