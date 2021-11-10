import csv
from statistics import mean 
import os

csv_path_CouplingMap_montreal=os.path.dirname(os.path.abspath(__file__))+"/test_CouplingMap_montreal/results/"
csv_path_CouplingMap_FullyConnected = os.path.dirname(os.path.abspath(__file__))+"/test_CouplingMap_FullyConnected/results/"

import numpy as np

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
        dict_data['num_qubits']=first_row[1]
        dict_data['depth']=int(mean_data(first_row[4]))
        return dict_data
def obtain_data_montreal_map(csv_file):
    with open(csv_path_CouplingMap_montreal+csv_file, newline='') as f:
        reader = csv.reader(f)
        headings = next(reader)
        #get the result of sabre
        first_row= next(reader)
        dict_data_sabre={}
        dict_data_sabre['num_qubits']=first_row[1]
        dict_data_sabre['depth']=int(mean_data(first_row[4]))
        #get the result of nassc
        second_row= next(reader)
        dict_data_nassc={}
        dict_data_nassc['num_qubits']=second_row[1]
        dict_data_nassc['depth']=int(mean_data(second_row[4]))
        return (dict_data_sabre,dict_data_nassc)


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
'''
labels = ['bv_n5', '3_17_13', 'mod5mils_65','decod24-v2_43', 'mod5d2_64', 'grover_n4',
          'grover_n6', 'grover_n8', 'vqe_n8', 'vqe_n12', 'bv_n19', 'qft_n15', 'qft_n20', 'qpe_n9',
          'adder_n10', 'multiplier_n25',
          'sqn_258', 'rd84_253', 'co14_215', 'sym9_193',
         ]      
'''   
labels = [#'bv_n5', '3_17_13', 'mod5mils_65','decod24-v2_43', 'mod5d2_64', 
        'grover_n4','grover_n6', 'grover_n8', 'vqe_n8', 'vqe_n12', 
        'bv_n19', 'qft_n15', 'qft_n20', 'qpe_n9','adder_n10', 'multiplier_n25',
        'sqn_258', 'rd84_253', 'co14_215', 'sym9_193',
        ]

with open('depth_table_using_montreal_map.csv', 'w') as csvfile:
    fieldnames = ['name', '#qubits', 'depth_total_OriginalCircuit',
                  'depth_total_Qiskit+SABRE', 'depth_add_Qiskit+SABRE', 
                  'depth_total_Qiskit+NASSC', 'depth_add_Qiskit+NASSC', 
                  'delta_depth_total', 'delta_depth_add', 
                 ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    delta_depth_total_list = []
    delta_depth_add_list = []
    
    for csv_file in labels:
        num_qubits= csv_file_dict_fully_connected_map[csv_file]['num_qubits']
        
        depth_total_OriginalCircuit = csv_file_dict_fully_connected_map[csv_file]['depth']
        depth_total_Qiskit_SABRE = csv_file_dict_montreal_map[csv_file][0]['depth']
        depth_total_Qiskit_NASSC = csv_file_dict_montreal_map[csv_file][1]['depth']
        
        depth_add_Qiskit_SABRE = depth_total_Qiskit_SABRE - depth_total_OriginalCircuit
        depth_add_Qiskit_NASSC = depth_total_Qiskit_NASSC - depth_total_OriginalCircuit

        delta_depth_total = (depth_total_Qiskit_SABRE - depth_total_Qiskit_NASSC)/depth_total_Qiskit_SABRE
        delta_depth_add = (depth_add_Qiskit_SABRE - depth_add_Qiskit_NASSC )/ depth_add_Qiskit_SABRE
        
        writer.writerow({'name': csv_file, '#qubits': num_qubits, 'depth_total_OriginalCircuit':depth_total_OriginalCircuit,
                        'depth_total_Qiskit+SABRE': depth_total_Qiskit_SABRE, 'depth_add_Qiskit+SABRE': depth_add_Qiskit_SABRE,
                        'depth_total_Qiskit+NASSC': depth_total_Qiskit_NASSC, 'depth_add_Qiskit+NASSC':depth_add_Qiskit_NASSC,
                        'delta_depth_total':"{:.2%}".format(delta_depth_total), 
                        'delta_depth_add':"{:.2%}".format(delta_depth_add), 
                        })
        delta_depth_total_list.append(delta_depth_total)
        delta_depth_add_list.append(delta_depth_add)
    
    writer.writerow({'name': 'geometric_mean',
                     'delta_depth_total':"{:.2%}".format(geometric_mean_data(delta_depth_total_list)), 
                     'delta_depth_add':"{:.2%}".format(geometric_mean_data(delta_depth_add_list)), 
                    })
