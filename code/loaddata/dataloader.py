# encoding: utf-8

"""
Read images and corresponding labels.
"""

import numpy as np
import os
import sys
import json
import torch
from torch.utils.data import Dataset

sys.path.append('loaddata')
import data_function


class DataSet(Dataset):
    def __init__(self, 
            patient_list, 
            patient_time_record_dict, 
            patient_label_dict,
            patient_master_dict,
            phase='train',          # phase
            split_num=5,            # split feature value into different parts
            args=None               # 全局参数
            ):

        self.patient_list = patient_list
        self.patient_time_record_dict = patient_time_record_dict
        self.patient_label_dict = patient_label_dict
        self.patient_master_dict = patient_master_dict
        self.phase = phase
        self.split_num = split_num
        self.split_nor = args.split_nor
        self.split_nn = args.split_nn
        self.args = args
        self.time_range = args.time_range
        self.last_time = args.last_time
        self.length = 24
        self.n_code = 8
        # self.feature_normal_range_order_dict = json.load(open('../result/phy_feature_normal_range_order_dict.json'))
        # self.index_feature_list = json.load(open('../result/index_feature_list.json'))

    def get_visit_info_wi_normal_range(self, time_record_dict):
        # times = sorted([float(t) for t in time_record_dict.keys()])
        times = sorted(time_record_dict.keys(), key=lambda s:float(s))
        max_time = float(times[-1])
        # print(times[-24:])
        # for t in time_record_dict:
        #     time_record_dict[str(float(t))] = time_record_dict[t]
        visit_list = []
        value_list = []
        mask_list = []
        time_list = []

        n_code = self.n_code
        import traceback

        # trend
        trend_list = []
        previous_value = [[[],[]] for _ in range(n_code)]
        change_th = 0.02
        start_time = - self.args.avg_time * 2
        end_time = -1

        feature_last_value = dict()
        init_data = []
        for time in times :
            if float(time) <= - self.time_range:
                continue
            if float(time) - max_time >= self.last_time * 60:
                continue
            time = str(time)
            records = time_record_dict[time].items()
            # records = time_record_dict[time]
            # print(records)
            feature_index = [int(r[0]) for r in records]
            feature_value = [float(r[1]) for r in records]

            # embed feature value
            feature_index = np.array(feature_index)
            feature_value = np.array(feature_value)
            # feature = feature_index * self.split_nn + feature_value * self.split_num
            feature = feature_index * self.split_nn

            # trend
            trend = np.zeros(n_code, dtype=np.int64)
            i_v = 0
            for idx, val in zip(feature_index, feature_value):
                # delete val with time less than start_time
                ptimes = previous_value[idx][0]
                lip = 0
                for ip, pt in enumerate(ptimes):
                    if pt >= float(time) + start_time:
                        lip = ip
                        break

                avg_val = None
                if len(previous_value[idx][0]) == 1:
                    avg_val = previous_value[idx][1][-1]

                previous_value[idx] = [
                        previous_value[idx][0][lip:],
                        previous_value[idx][1][lip:]]

                # trend value
                if len(previous_value[idx][0]):
                    avg_val = np.mean(previous_value[idx][1])
                if avg_val is not None:
                    if val < avg_val - change_th:
                        delta = 0
                    elif val > avg_val + change_th:
                        delta = 1
                    else:
                        delta = 2
                    trend[i_v] = idx * 3 + delta + 1

                # add new val
                previous_value[idx][0].append(float(time))
                previous_value[idx][1].append(float(val))

                i_v += 1





            visit = np.zeros(n_code, dtype=np.int64)
            mask = np.zeros(n_code, dtype=np.int64)
            i_v = 0
            for feat, idx, val in zip(feature, feature_index,  feature_value):

                # order
                mask[i_v] = 1
                visit[i_v] = int(feat + 1)
                # normal_range = self.feature_normal_range_order_dict[self.index_feature_list[idx]]
                normal_range = [0.4, 0.6]

                range_value = 0
                if val < normal_range[0]:
                    if normal_range[0]  > 0.1:
                        if val > normal_range[0]/2:
                            range_value += 1
                elif val > normal_range[1]:
                    range_value += 3
                    if normal_range[1]  < 0.9:
                        if 1 - val <  (1 - normal_range[1]) / 2:
                            range_value += 1
                else:
                    range_value += 2

                delta_value = 1
                if self.args.use_trend:
                    if idx in feature_last_value:
                        last_value = feature_last_value[idx]
                        delta = 0.3
                        if val - last_value < - delta:
                            delta_value = 0
                            # print(err)
                            feature_last_value[idx] = val
                        elif val - last_value > delta:
                            delta_value = 2
                            # print(err)
                            feature_last_value[idx] = val
                    else:
                        feature_last_value[idx] = val

                visit[i_v] += range_value * 3 + delta_value

                i_v += 1
                    

            value = np.zeros((2, n_code ), dtype=np.int64)
            value[0][: len(feature_index)] = feature_index + 1
            value[1][: len(feature_index)] = (feature_value * self.args.n_split).astype(np.int64)
            value_list.append(value)

            visit_list.append(visit)
            mask_list.append(mask)
            time_list.append(float(time))
            trend_list.append(trend)
            init_data.append(dict(records))

        num_len = self.length 

        if len(visit_list) <= num_len:
            visit = np.zeros(n_code, dtype=np.int64)
            trend = np.zeros(n_code, dtype=np.int64)
            value = np.zeros((2, n_code), dtype=np.int64)
            while len(visit_list) < num_len:
                visit_list.append(visit)
                value_list.append(value)
                mask_list.append(visit)
                time_list.append(0)
                trend_list.append(trend)
                init_data.append([])
        else:
            visit_list = visit_list[- self.length:]
            value_list = value_list[- self.length:]
            mask_list = mask_list[- self.length:]
            time_list = time_list[- self.length:]
            trend_list = trend_list[- self.length:]
            init_data = init_data[- self.length :]


        return np.array(visit_list), np.array(value_list), np.array(mask_list, dtype=np.float32), np.array(time_list, dtype=np.float32), np.array(trend_list), init_data




    def get_visit_info_wo_normal_range(self, time_record_dict):
        # times = sorted([float(t) for t in time_record_dict.keys()])
        times = sorted(time_record_dict.keys(), key=lambda s:float(s))
        # for t in time_record_dict:
        #     time_record_dict[str(float(t))] = time_record_dict[t]
        visit_list = []
        value_list = []
        mask_list = []
        time_list = []

        n_code = self.n_code
        import traceback

        # trend
        trend_list = []
        previous_value = [[[],[]] for _ in range(n_code)]
        change_th = 0.02
        start_time = - self.args.avg_time * 2
        end_time = -1

        for time in times :
            if float(time) <= - self.time_range:
                continue
            if float(time) >= self.last_time:
                continue
            time = str(time)
            records = time_record_dict[time].items()
            # print (records)
            feature_index = [int(r[0]) for r in records]
            feature_value = [float(r[1]) for r in records]

            # embed feature value
            feature_index = np.array(feature_index)
            feature_value = np.array(feature_value)
            feature = feature_index * self.split_nn + feature_value * self.split_num

            # trend
            trend = np.zeros(n_code, dtype=np.int64)
            i_v = 0
            for idx, val in zip(feature_index, feature_value):
                # delete val with time less than start_time
                ptimes = previous_value[idx][0]
                lip = 0
                for ip, pt in enumerate(ptimes):
                    if pt >= float(time) + start_time:
                        lip = ip
                        break

                avg_val = None
                if len(previous_value[idx][0]) == 1:
                    avg_val = previous_value[idx][1][-1]

                previous_value[idx] = [
                        previous_value[idx][0][lip:],
                        previous_value[idx][1][lip:]]

                # trend value
                if len(previous_value[idx][0]):
                    avg_val = np.mean(previous_value[idx][1])
                if avg_val is not None:
                    if val < avg_val - change_th:
                        delta = 0
                    elif val > avg_val + change_th:
                        delta = 1
                    else:
                        delta = 2
                    trend[i_v] = idx * 3 + delta + 1

                # add new val
                previous_value[idx][0].append(float(time))
                previous_value[idx][1].append(float(val))

                i_v += 1





            visit = np.zeros(n_code, dtype=np.int64)
            mask = np.zeros(n_code, dtype=np.int64)
            i_v = 0
            for feat, idx, val in zip(feature, feature_index,  feature_value):

                # order
                mask[i_v] = 1
                visit[i_v] = int(feat + 1)
                i_v += 1


                    

            value = np.zeros((2, n_code ), dtype=np.int64)
            value[0][: len(feature_index)] = feature_index + 1
            value[1][: len(feature_index)] = (feature_value * self.args.n_split).astype(np.int64)
            value_list.append(value)

            visit_list.append(visit)
            mask_list.append(mask)
            time_list.append(float(time))
            trend_list.append(trend)

        num_len = self.length 

        if len(visit_list) <= num_len:
            visit = np.zeros(n_code, dtype=np.int64)
            trend = np.zeros(n_code, dtype=np.int64)
            value = np.zeros((2, n_code), dtype=np.int64)
            while len(visit_list) < num_len:
                visit_list.append(visit)
                value_list.append(value)
                mask_list.append(visit)
                time_list.append(0)
                trend_list.append(trend)
        else:
            visit_list = visit_list[- self.length:]
            value_list = value_list[- self.length:]
            mask_list = mask_list[- self.length:]
            time_list = time_list[- self.length:]
            trend_list = trend_list[- self.length:]


        return np.array(visit_list), np.array(value_list), np.array(mask_list, dtype=np.float32), np.array(time_list, dtype=np.float32), np.array(trend_list)




    def __getitem__(self, index):
        patient = self.patient_list[index]
        if self.args.use_visit:
            visit_list, value_list, mask_list, time_list, trend_list,  init_data= self.get_visit_info_wi_normal_range(self.patient_time_record_dict[patient])
            v = value_list[:, 0, :]
            # print(v.min(), v.max())
            v = value_list[:, 1, :]
            # print(v.min(), v.max())
            # print('----')
            # print('----')
            if os.path.exists(self.args.master_file):
                master = self.patient_master_dict[patient]
                master = [int(m) for m in master]
                master = np.float32(master)
            else:
                master = 0
            if self.args.final == 1:
                label = np.float32(0)
            else:
                label = np.float32(self.patient_label_dict[patient])

            if self.args.compute_weight and self.phase != 'train' and label>0:
                with open(os.path.join(self.args.result_dir, 'cr', patient + '.init.json'), 'w') as f:
                    f.write(json.dumps(init_data, indent=4))
            if visit_list.max() > 121:
                # print(visit_list.max())
                pass
            if self.phase == 'test':
                return visit_list, value_list, mask_list, master, label, time_list, trend_list, patient
            else:
                return visit_list, value_list, mask_list, master, label, time_list, trend_list, patient




    def __len__(self):
        return len(self.patient_list) 
