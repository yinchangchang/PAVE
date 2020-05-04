# coding=utf8


'''
main.py 为程序入口
'''


# 基本依赖包
import os
import sys
import time
import json
import traceback
import numpy as np
from glob import glob
from tqdm import tqdm
from tools import parse, py_op


# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


# 自定义文件
import loss
import models
import function
import loaddata
# import framework
from loaddata import dataloader
from models import attention


# 全局变量
args = parse.args
args.hard_mining = 0
if torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = 0

if args.model == 'attention':
    args.epochs = max(30, args.epochs)
    
args.use_trend = max(args.use_trend, args.use_value)
args.use_value = max(args.use_trend, args.use_value)
args.rnn_size = args.embed_size
args.hidden_size = args.embed_size
if args.num_layers > 1 or args.model != 'attention':
    args.compute_weight = 0
args.compute_weight = 0

def my_tqdm(x):
    tqdm(x)
    return x


def train_eval(p_dict, phase='train'):
    print(args.model)
    ### 传入参数
    epoch = p_dict['epoch']
    model = p_dict['model']           # 模型
    loss = p_dict['loss']             # loss 函数
    if phase == 'train':
        data_loader = p_dict['train_loader']        # 训练数据
        optimizer = p_dict['optimizer']             # 优化器
        model.train()
    else:
        data_loader = p_dict['val_loader']
        model.eval()

    ### 局部变量定义
    classification_metric_dict = dict()

    for i,data in enumerate(tqdm(data_loader)):
        if args.use_visit:
            names = data[-1]
            data = data[:-1]
            if args.gpu:
                data = [ Variable(x.cuda()) for x in data ]
            visits, values, mask, master, labels, times, trends = data
            if i == 0:
                print('input size', visits.size()) 
                print('Time size', times.size()) 
                print('demo size', master.size()) 
            output_list = model(visits, master, mask, times, phase, values, trends)
            output = output_list[0]

        classification_loss_output = loss(output, labels, args.hard_mining)
        loss_gradient = classification_loss_output[0]
        # 计算性能指标
        function.compute_metric(output, labels, time, classification_loss_output, classification_metric_dict, phase)


        # 训练阶段
        if phase == 'train':
            optimizer.zero_grad()
            loss_gradient.backward()
            optimizer.step()

        # if i >= 10:
        #     break


    print('\nEpoch: {:d} \t Phase: {:s} \n'.format(epoch, phase))
    metric = function.print_metric('classification', classification_metric_dict, phase)
    if args.phase != 'train':
        print ('metric = ', metric)
        print()
        print()
    if phase == 'val':
        if metric > p_dict['best_metric'][0]:
            p_dict['best_metric'] = [metric, epoch]
            function.save_model(p_dict)

        print('valid: metric: {:3.4f}\t epoch: {:d}\n'.format(metric, epoch))
        print('\t\t\t valid: best_metric: {:3.4f}\t epoch: {:d}\n'.format(p_dict['best_metric'][0], p_dict['best_metric'][1]))  
    else:
        print('train: metric: {:3.4f}\t epoch: {:d}\n'.format(metric, epoch))



def main():
    p_dict = dict() # All the parameters
    p_dict['args'] = args
    args.split_nn =  3 * 5
    args.vocab_size = args.split_nn * 145 + 2
    print ('vocab_size', args.vocab_size)

    ### load data
    print ('read data ...')
    if args.task == 'mortality':

        patient_time_record_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_time_record_dict.json'))
        patient_master_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_master_dict.json'))
        patient_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_label_dict.json'))

    
        if os.path.exists(os.path.join(args.result_dir, 'train.json')):
            patient_train = list(json.load(open(os.path.join(args.result_dir, 'train.json'))))
            patient_valid = list(json.load(open(os.path.join(args.result_dir, 'valid.json')))) 
            patient_test = list(json.load(open(os.path.join(args.result_dir, 'test.json')))) 
        else:
            patients = sorted(set(patient_label_dict.keys()) & set(patient_time_record_dict) & set(patient_master_dict))
            print(len(patient_master_dict), len(patient_label_dict), len(patient_time_record_dict))
            print('There are {:d} patients.'.format(len(patients)))
            n_train = int(0.7 * len(patients))
            n_valid = int(0.2 * len(patients))
            patient_train = patients[:n_train]
            patient_valid = patients[n_train:n_train+n_valid]
            patient_test  = patients[n_train+n_valid:]

        args.master_size = len(patient_master_dict[patients[0]])
    elif args.task == 'sepsis':
        patient_time_record_dict = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_time_record_dict.json'))
        patient_master_dict = py_op.myreadjson(os.path.join(args.result_dir, 'patient_master_dict.json'))
        patient_label_dict = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_label_dict.json'))
        sepsis_split = py_op.myreadjson(os.path.join(args.result_dir, 'sepsis_split.json'))
        print(sepsis_split.keys())
        sepsis_split = sepsis_split[str(- args.last_time)]
    
        patient_train = sepsis_split['train']
        patient_valid = sepsis_split['valid']
        print('train: {:d}'.format(len(patient_train)))
        print('valid: {:d}'.format(len(patient_valid)))






    print ('data loading ...')
    train_dataset  = dataloader.DataSet(
                patient_train, 
                patient_time_record_dict,
                patient_label_dict,
                patient_master_dict, 
                args=args,
                phase='train')
    train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)
    val_dataset  = dataloader.DataSet(
                patient_valid, 
                patient_time_record_dict,
                patient_label_dict,
                patient_master_dict, 
                args=args,
                phase='val')
    val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)
    test_dataset  = dataloader.DataSet(
                patient_test, 
                patient_time_record_dict,
                patient_label_dict,
                patient_master_dict, 
                args=args,
                phase='val')
    test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)

    p_dict['train_loader'] = train_loader
    if args.phase == 'train':
        p_dict['val_loader'] = val_loader
    else:
        p_dict['val_loader'] = test_loader



    cudnn.benchmark = True
    net = attention.Attention(args)
    if args.gpu:
        net = net.cuda()
        p_dict['loss'] = loss.Loss().cuda()
    else:
        p_dict['loss'] = loss.Loss()

    parameters = []
    for p in net.parameters():
        parameters.append(p)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    p_dict['optimizer'] = optimizer
    p_dict['model'] = net
    start_epoch = 0
    # args.epoch = start_epoch
    # print ('best_f1score' + str(best_f1score))

    p_dict['epoch'] = 0
    p_dict['best_metric'] = [0, 0]


    ### resume pretrained model
    if os.path.exists(args.resume):
        print ('resume from model ' + args.resume)
        function.load_model(p_dict, args.resume)
        print ('best_metric', p_dict['best_metric'])


    if args.phase == 'train':

        best_f1score = 0
        for epoch in range(p_dict['epoch'] + 1, args.epochs):
            p_dict['epoch'] = epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            train_eval(p_dict, 'train')
            train_eval(p_dict, 'val')
        log_info = '# task : {:s}; model: {:s} ; last_time: {:d} ; auc: {:3.4f} \n'.format(args.task, args.model, args.last_time, p_dict['best_metric'][0])
        with open('../result/log.txt', 'a') as f:
            f.write(log_info)
    else:
        train_eval(p_dict, 'test')


if __name__ == '__main__':
    main()
