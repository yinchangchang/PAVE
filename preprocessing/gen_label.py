# coding=utf8

import os
import sys
import json
import time
import pandas as pd
sys.path.append('../code')

import tools
from tools import parse, py_op
args = parse.args

def time_to_min(t):
    t = t.replace('"', '')
    # t = time.mktime(time.strptime(t.replace('"', '')))
    t = time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S'))
    t = t / 60
    return int(t)



def gen_patient_label_dict():
    patient_label_dict = dict()
    label_file = args.label_file
    for i_line,line in enumerate(open(label_file)):
        if i_line != 0:
            data = line.strip().split(',')
            patient = str(int(float(data[0])))
            # patient = data[0]
            label  = data[-1]
            patient_label_dict[patient] = int(float(label))
    py_op.mywritejson(os.path.join(args.result_dir, 'patient_label_dict.json'), patient_label_dict)

    print('There are {:d} positive samples.'.format(sum(patient_label_dict.values())))
    print('There are {:d} negtive samples.'.format(len(patient_label_dict) - sum(patient_label_dict.values())))

def gen_sepsis_label_dict():
    sepsis_label_dict = dict()

    sepsis_file  = '../data/sepsis3.csv'
    print('reading sepsis3.csv')
    sepsis_data = pd.read_csv(sepsis_file)
    sepsis_infection_dict = dict()
    sepsis_set = set()
    for iline in range(len(sepsis_data)):
        adm = sepsis_data.loc[iline, 'hadm_id']
        adm = str(adm)
        excluded =  sepsis_data.loc[iline, 'excluded']
        suspected_infection_time_poe =  sepsis_data.loc[iline, 'suspected_infection_time_poe']
        if len(str(suspected_infection_time_poe)) > 5:
            sepsis_infection_dict[adm] = time_to_min(suspected_infection_time_poe)
            # sepsis_set.add(adm)
            if excluded==0: 
                sepsis_set.add(adm)
        # if len(str(suspected_infection_time_poe)) > 5:
            # sepsis_infection_dict[adm] = time_to_min(suspected_infection_time_poe)
        # print(suspected_infection_time_poe)
        # if excluded == 0 and len(str(suspected_infection_time_poe)) > 0:
        #     sepsis_set.add(adm)
        #     return
    # print(len(sepsis_infection_dict))
    # print(len(sepsis_set))
    print('Infection No: {:d}'.format(len(sepsis_infection_dict)))
    print('Sepsis No: {:d}'.format(len(sepsis_set)))
    # py_op.mywritejson(os.path.join(args.result_dir, 'sepsis_time_dict.json'), patient_label_dict)
    # return

    icu_file  = '../data/icustays.csv'
    print('reading icustays.csv')
    icu_data = pd.read_csv(icu_file)
    icu_adm_dict = dict()
    for iline in range(len(icu_data)):
        icu = icu_data.loc[iline, 'icustay_id']
        adm = icu_data.loc[iline, 'hadm_id']
        icu_adm_dict[icu] = adm

    sofa_file = '../data/sofa.csv'
    print('reading sofa.csv')
    sofa_data = pd.read_csv(sofa_file)


    print('mapping sofa to adm')
    adm_sofa_dict = dict()
    for iline in range(len(sofa_data)):
        break
        if iline and iline % 10000 == 0:
            print('mapping sofa to adm', iline, len(sofa_data))
        icu = sofa_data.loc[iline, 'icustay_id']
        sofa = sofa_data.loc[iline, 'sofa_24hours']
        starttime = sofa_data.loc[iline, 'starttime']
        endtime = sofa_data.loc[iline, 'endtime']
        adm = icu_adm_dict[icu]
        adm_sofa_dict[adm] = adm_sofa_dict.get(adm , []) + [[sofa, starttime, endtime]]
    # py_op.mywritejson('../result/adm_sofa_dict.json', adm_sofa_dict)
    # return
    adm_sofa_dict = py_op.myreadjson('../result/adm_sofa_dict.json')
    
    
    print('set sepsis label')
    pos_num = 0
    for iline,(adm, sofa_list) in enumerate(adm_sofa_dict.items()):
        # print(adm, type(adm))
        if iline and iline % 10000 == 0:
            print('set sepsis label', iline, len(adm_sofa_dict))
        # if adm not in sepsis_infection_dict:
        if adm in sepsis_infection_dict:
            sepsis_label_dict[adm] = [0, sepsis_infection_dict[adm]]
        else:
            continue
        if adm not in sepsis_set:
            continue

        # sofa_list = sofa_list

        # if time_to_min(sofa_list[0][1]) < sepsis_infection_dict[adm] :
        #     continue

        # print('have data')

        sofa_init = ''
        for sofa in sofa_list:
            starttime = sofa[1]
            endtime = sofa[2]
            time = time_to_min(endtime)
            sofa = int(sofa[0])
            if time - sepsis_infection_dict[adm] >= - 48*60 and time - sepsis_infection_dict[adm] <= 24*60:
                if sofa_init == '':
                    sofa_init = sofa
                elif sofa - sofa_init >=  2 and sofa >=2:
                    sepsis_label_dict[adm] = [1, sepsis_infection_dict[adm]]
                    sepsis_infection_dict[adm] = max(time, sepsis_infection_dict[adm])
                    pos_num += 1
                    break


    
    print('writing sepsis_label_dict')
    py_op.mywritejson(os.path.join(args.result_dir, 'sepsis_time_dict.json'), sepsis_infection_dict)
    py_op.mywritejson(os.path.join(args.result_dir, 'sepsis_label_dict.json'), { k:v[0] for k,v in sepsis_label_dict.items() })

    print('There are {:d} positive samples.'.format(pos_num))
    print('There are {:d} negtive samples.'.format(len(sepsis_label_dict) - pos_num))

def compare_sepsis():
    print('reading')
    sepsis_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_label_dict.json'))
    print('reading')
    patient_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_label_dict.json'))
    print(len(set(sepsis_label_dict) & set(patient_label_dict)))
    # sepsis_label_dict = [k for k,v in sepsis_label_dict.items() if v ]
    print(len(set(sepsis_label_dict) & set(patient_label_dict)))
    d = dict()
    for p,l in sepsis_label_dict.items():
        if p not in patient_label_dict:
            continue
        if l == 0:
            d[p] = 0
        else:
            d[p] = 1
    print(len(d))
    print(sum(d.values()))
    py_op.mywritejson(os.path.join(args.result_dir, 'sepsis_label_dict.json'), d)

    sepsis_time_dict = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_time_dict.json'))
    sepsis_time_dict = { k:v for k,v in sepsis_time_dict.items() if k in d}
    print(len(sepsis_time_dict))
    py_op.mywritejson(os.path.join(args.result_dir, 'sepsis_time_dict.json'), sepsis_time_dict)

def split_data():
    patient_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_label_dict.json'))
    # patients = patient_label_dict.keys()
    # patients = sorted(patients)
    patients = py_op.myreadjson(os.path.join(args.result_dir, 'patient_list.json'))
    n = int(len(patients) * 0.8)
    patient_train = patients[:n]
    patient_valid = patients[n:]
    py_op.mywritejson(os.path.join(args.result_dir, 'train.json'), patient_train)
    py_op.mywritejson(os.path.join(args.result_dir, 'valid.json'), patient_valid)
    print(sum([patient_label_dict[k] for k in patient_train])) 
    print(sum([patient_label_dict[k] for k in patient_valid])) 
    print(len([patient_label_dict[k] for k in patient_train])) 







def main():
    gen_patient_label_dict()
    # split_data()
    # gen_sepsis_label_dict()
    # compare_sepsis()
    # split_data()





if __name__ == '__main__':
    main()
