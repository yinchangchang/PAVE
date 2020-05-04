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



def gen_json_data():
    vital_file = args.vital_file
    patient_time_record_dict = dict()
    print('reading...')
    feature_index_dict = py_op.myreadjson(os.path.join(args.result_dir, 'feature_index_dict.json'))
    index_feature_list = py_op.myreadjson(os.path.join(args.result_dir, 'index_feature_list.json'))
    feature_value_order_dict = py_op.myreadjson(os.path.join(args.result_dir, 'feature_value_order_dict.json'))
    feature_value_order_dict = { str(feature_index_dict[k]):v for k,v in feature_value_order_dict.items()  if 'time' not in k}
    patient_time_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_time_dict.json'))
    print('end reading')
    for i_line, line in enumerate(open(vital_file)):
        if i_line and i_line % 10000 == 0:
            print('line', i_line) 
        if i_line:
            data = line.strip().split(',')
            patient, time = data[:2]
            time = time_to_min(time)
            time = int(float(time))
            if patient not in patient_time_record_dict:
                patient_time_record_dict[patient] = dict()

            data = data[2:]
            vs = dict()
            for idx, val in enumerate(data):
                if len(val) == 0:
                    continue
                value_order = feature_value_order_dict[str(idx)]
                vs[idx] = float('{:3.3f}'.format(value_order[val]))
            patient_time_record_dict[patient][time - patient_time_dict[patient] - 1] = vs

    with open(os.path.join(args.result_dir, 'patient_time_record_dict.json'), 'w') as f:
        f.write(json.dumps(patient_time_record_dict))

def gen_sepsis_json_data(): 
    vital_file = args.vital_file
    patient_time_record_dict = dict()
    feature_index_dict = py_op.myreadjson(os.path.join(args.result_dir, 'feature_index_dict.json'))
    index_feature_list = py_op.myreadjson(os.path.join(args.result_dir, 'index_feature_list.json'))
    feature_value_order_dict = py_op.myreadjson(os.path.join(args.result_dir, 'feature_value_order_dict.json'))
    feature_value_order_dict = { str(feature_index_dict[k]):v for k,v in feature_value_order_dict.items()  if 'time' not in k}
    patient_time_dict = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_time_dict.json'))
    # return
    for i_line, line in enumerate(open(vital_file)):
        if i_line and i_line % 10000 == 0:
            print('line', i_line) 
        if i_line:
            data = line.strip().split(',')
            patient, time = data[:2]
            time = time_to_min(time)
            if patient not in patient_time_dict:
                continue
            if time > patient_time_dict[patient]:
                continue
            time = int(float(time))
            if patient not in patient_time_record_dict:
                patient_time_record_dict[patient] = dict()

            data = data[2:]
            vs = dict()
            for idx, val in enumerate(data):
                if len(val) == 0:
                    continue
                value_order = feature_value_order_dict[str(idx)]
                vs[idx] = float('{:3.3f}'.format(value_order[val]))
            patient_time_record_dict[patient][time - patient_time_dict[patient] - 1] = vs

    with open(os.path.join(args.result_dir, 'sepsis_time_record_dict.json'), 'w') as f:
        f.write(json.dumps(patient_time_record_dict))


def analyze_sepsis():
    sepsis_time_record_dict = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_time_record_dict.json'))
    sepsis_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_label_dict.json'))
    print(len(sepsis_time_record_dict))
    n = 0
    np = 0
    d = {
            30: 0,
            60: 0,
            120: 0,

            }
    for p,vd in sepsis_time_record_dict.items():
        if sepsis_label_dict[p]:
            n += 1
        else:
            continue
            pass
        min_t = - int(min(vd.keys()))
        for k in d:
            if min_t < k:
                d[k] += 1
    print(n)
    print(d)
    sepsis_label_dict = { k:v for k,v in sepsis_label_dict.items() if k in sepsis_time_record_dict }
    # py_op.mywritejson(os.path.join(args.result_dir, 'sepsis_label_dict.json'), sepsis_label_dict)


def feature_change():
    print('reading')
    patient_time_record_dict = json.load(open(os.path.join(args.result_dir, 'patient_time_record_dict.json')))
    print(patient_time_record_dict.keys()) 
    patient_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_label_dict.json'))
    feature_list_dict = { str(i): [] for i in range(8) }
    for p in patient_time_record_dict:
        if p in patient_label_dict and patient_label_dict[p]:
            tr = patient_time_record_dict[p]
            last_v = { }
            for rs in tr.values():
                for i,v in rs.items():
                    if i in last_v:
                        feature_list_dict[i].append(abs(v - last_v[i]))
                    last_v[i] = v
    for f,l in feature_list_dict.items():
        l = sorted(l)
        ds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
        ds = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
        ns = []
        for i,delta in enumerate(l):
            if delta  > ds[0]:
                ns.append(float('{:3.2f}'.format(1.0*i/len(l))))
                ds = ds[1:]
        ns.append(len(l))
        print(f, ns) 

def get_cases():
    sepsis_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_label_dict.json'))
    print(len(sepsis_label_dict))
    icu_file  = '../data/icustays.csv'
    print('reading icustays.csv')
    icu_data = pd.read_csv(icu_file)
    icu_adm_dict = dict()
    icu_intime_dict = dict()
    for iline in range(len(icu_data)):
        icu = icu_data.loc[iline, 'icustay_id']
        intime = icu_data.loc[iline, 'intime']
        adm = icu_data.loc[iline, 'hadm_id']
        icu_adm_dict[icu] = adm
        icu_intime_dict[icu] = time_to_min(intime)

    sepsis_label_dict = { k:0 for k in sepsis_label_dict }
    sepsis_time_dict = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_time_dict.json'))

    for iline, line in enumerate(open('../data/sepsis_onset_time.csv')):
        icustay_id, h = line.strip().split(',')
        adm = icu_adm_dict[int(icustay_id)]
        sepsis_label_dict[adm] = 1
        time = icu_intime_dict[int(icustay_id)] + 60 * int(h)
        sepsis_time_dict[adm] = time


    for iline, line in enumerate(open('../data/sepsis3_cases.csv')):
        break
        if iline:
            icustay_id,intime,outtime,length_of_stay,delta_score,sepsis_onset,sepsis_onset_day,sepsis_onset_hour = line.strip().split(',')
            adm = icu_adm_dict[int(icustay_id)]
            sepsis_label_dict[adm] = 1

            time = time_to_min(sepsis_onset)
            sepsis_time_dict[adm] = time

    py_op.mywritejson(os.path.join(args.result_dir, 'sepsis_label_dict.json'), sepsis_label_dict)
    py_op.mywritejson(os.path.join(args.result_dir, 'sepsis_time_dict.json'), sepsis_time_dict)





def main():
    gen_json_data()

    # sepsis
    # get_cases()
    # gen_sepsis_json_data()

    # analyze_sepsis()
    # feature_change()











if __name__ == '__main__':
    main()
