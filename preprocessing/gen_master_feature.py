# coding=utf8

import os
import sys
sys.path.append('../code')

import tools
from tools import parse, py_op
args = parse.args


def gen_master_feature_list():


    # master information
    master_file = args.master_file
    m_set = set()
    for i_line,line in enumerate(open(master_file)):
        if i_line != 0:
            data = line.strip().split(',')
            age = float(data[2])
            age = str(int(age / 10))
            data[2] = age
            for i,d in enumerate(data[1:]):
                m_set.add(str(i) + d)
    return sorted(m_set)



def gen_patient_master_dict(master_list):
    patient_master_dict = dict()
    # master information
    master_file = args.master_file
    master_set = [set() for _ in range(6)]
    for i_line,line in enumerate(open(master_file)):
        if i_line != 0:
            data = line.strip().split(',')
            patient = data[0]
            feature = ['0' for _ in range(len(master_list))]
            for i,d in enumerate(data[1:]):
                if i == 1:
                    d = str(int(float(d)/10))
                m = str(i) + d
                idx = master_list.index(m)
                feature[idx] = '1'
            patient_master_dict[patient] = ''.join(feature)
    py_op.mywritejson(os.path.join(args.result_dir, 'patient_master_dict.json'), patient_master_dict)

def main():
    master_list = gen_master_feature_list()
    print(len(master_list))
    gen_patient_master_dict(master_list)





if __name__ == '__main__':
    main()
