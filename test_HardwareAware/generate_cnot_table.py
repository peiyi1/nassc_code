import csv
import os
from  statistics import mean
from qiskit.visualization import plot_histogram

csv_path_CouplingMap_montreal=os.path.dirname(os.path.abspath(__file__))+"/test_CouplingMap_montreal/results/"
csv_path_CouplingMap_FullyConnected = os.path.dirname(os.path.abspath(__file__))+"/test_CouplingMap_FullyConnected/results/"

import numpy as np
import matplotlib.pyplot as plt

def geometric_mean(data):
    x=np.log(data)
    return np.exp(x.mean())

def geometric_mean_data(data):
    has_negative_data = False
    result = 0
    for i in data:
        if i <= 0:
            has_negative_data = True
            break
    if has_negative_data:
        positive_list=[]
        for i in data:
            positive_list.append(i+1)
        result = geometric_mean(positive_list) - 1
    else:
        result = geometric_mean(data)
    return  result

def mean_data(data):
    return mean([float(i) for i in data.strip('][').split(', ')])

def obtain_data_fully_connected_map(csv_file):
    with open(csv_path_CouplingMap_FullyConnected +csv_file, newline='') as f:
        reader = csv.reader(f)
        headings = next(reader)
        #get the result of sabre
        first_row= next(reader)
        dict_data={}
        dict_data['num_qubits']=first_row[0]
        dict_data['num_cnot']=int(mean_data(first_row[1]))
        dict_data['depth']=int(mean_data(first_row[2]))
        return dict_data
def obtain_data_montreal_map(csv_file):
    with open(csv_path_CouplingMap_montreal+csv_file, newline='') as f:
        reader = csv.reader(f)
        headings = next(reader)
        #get the result of sabre
        first_row= next(reader)
        dict_data_sabre={}
        dict_data_sabre['num_qubits']=first_row[1]
        dict_data_sabre['num_cnot']=int(mean_data(first_row[2]))
        dict_data_sabre['fidelity']=mean_data(first_row[3])
        dict_data_sabre['depth']=int(mean_data(first_row[4]))
        #get the result of nassc
        second_row= next(reader)
        dict_data_nassc={}
        dict_data_nassc['num_qubits']=second_row[1]
        dict_data_nassc['num_cnot']=int(mean_data(second_row[2]))
        dict_data_nassc['fidelity']=mean_data(second_row[3])
        dict_data_nassc['depth']=int(mean_data(second_row[4]))
        #get the result of sabre_HardwareAware
        third_row= next(reader)
        dict_data_sabre_HardwareAware={}
        dict_data_sabre_HardwareAware['num_qubits']=third_row[1]
        dict_data_sabre_HardwareAware['num_cnot']=int(mean_data(third_row[2]))
        dict_data_sabre_HardwareAware['fidelity']=mean_data(third_row[3])
        dict_data_sabre_HardwareAware['depth']=int(mean_data(third_row[4]))
        #get the result of nassc_HardwareAware
        fouth_row= next(reader)
        dict_data_nassc_HardwareAware={}
        dict_data_nassc_HardwareAware['num_qubits']=fouth_row[1]
        dict_data_nassc_HardwareAware['num_cnot']=int(mean_data(fouth_row[2]))
        dict_data_nassc_HardwareAware['fidelity']=mean_data(fouth_row[3])
        dict_data_nassc_HardwareAware['depth']=int(mean_data(fouth_row[4]))
        return (dict_data_sabre,dict_data_nassc,dict_data_sabre_HardwareAware, dict_data_nassc_HardwareAware)


csv_file_set=set()
for csv_file in os.listdir(csv_path_CouplingMap_montreal):
    name,csv_fomat=csv_file.split('.')
    if(len(name)):
        csv_file_set.add(name)
csv_file_dict_fully_connected_map= dict.fromkeys(csv_file_set,None)
csv_file_dict_montreal_map= dict.fromkeys(csv_file_set,None)

for csv_file in csv_file_dict_fully_connected_map:
    csv_file_dict_fully_connected_map[csv_file]=obtain_data_fully_connected_map(csv_file+'.csv')

for csv_file in csv_file_dict_montreal_map:
    csv_file_dict_montreal_map[csv_file]=obtain_data_montreal_map(csv_file+'.csv')

#labels = list(csv_file_set)

num_cnot_add_sabre = []
num_cnot_add_nassc = []
num_cnot_add_sabre_hardwareaware = []
num_cnot_add_nassc_hardwareaware = []

labels = ['bv_n5', '3_17_13', 'mod5mils_65','decod24-v2_43', 'mod5d2_64', 'grover_n4']

for csv_file in labels:
    total_OriginalCircuit = csv_file_dict_fully_connected_map[csv_file]['num_cnot']
    total_Qiskit_SABRE = csv_file_dict_montreal_map[csv_file][0]['num_cnot']
    total_Qiskit_NASSC = csv_file_dict_montreal_map[csv_file][1]['num_cnot']
    total_Qiskit_SABRE_HardwareAware = csv_file_dict_montreal_map[csv_file][2]['num_cnot']
    total_Qiskit_NASSC_HardwareAware = csv_file_dict_montreal_map[csv_file][3]['num_cnot']

    add_Qiskit_SABRE = total_Qiskit_SABRE - total_OriginalCircuit
    add_Qiskit_NASSC = total_Qiskit_NASSC - total_OriginalCircuit
    add_Qiskit_SABRE_HardwareAware = total_Qiskit_SABRE_HardwareAware - total_OriginalCircuit
    add_Qiskit_NASSC_HardwareAware = total_Qiskit_NASSC_HardwareAware - total_OriginalCircuit

    num_cnot_add_sabre.append(add_Qiskit_SABRE)
    num_cnot_add_nassc.append(add_Qiskit_NASSC)
    num_cnot_add_sabre_hardwareaware.append(add_Qiskit_SABRE_HardwareAware)
    num_cnot_add_nassc_hardwareaware.append(add_Qiskit_NASSC_HardwareAware)

x = np.arange(len(labels)) 
width = 0.6 

fig, ax = plt.subplots()
rects0 = ax.bar(x , num_cnot_add_sabre, width/4, label='sabre',color='tab:blue')
rects1 = ax.bar(x + width/4, num_cnot_add_nassc, width/4, label='nassc',color='tab:orange')
rects2 = ax.bar(x + width/2, num_cnot_add_sabre_hardwareaware, width/4, label='sabre_HardwareAware',color='tab:gray')
rects3 = ax.bar(x + width/4*3, num_cnot_add_nassc_hardwareaware, width/4, label='nassc_HardwareAware',color='tab:red')
ax.set_ylabel('additional cnot')
#ax.set_title('')
ax.set_xticks(x+width/4)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.xticks(x, labels, rotation=45)
plt.savefig('cnot_compare.pdf', bbox_inches = 'tight')
