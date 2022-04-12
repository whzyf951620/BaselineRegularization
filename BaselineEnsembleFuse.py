import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import glob
import random
import time
from functools import reduce

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file
def ListAdd(a, b):
    assert type(a) == list and type(b) == list
    assert len(a) == len(b)
    index = 0
    while index < len(a):
        a[index] += b[index]
        a[index] = a[index] / 2
        index += 1

    return a

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    scores = nn.Softmax(dim = 1)(scores)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100
    return acc

def feature_evaluation_MultiModels(cl_data_fileList, modelList, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_fileList[0].keys()

    select_class = random.sample(class_list,n_way)
    z_allList = [[], [], []]
    for cl in select_class:
        img_featList = [cl_data_fileList[0][cl], cl_data_fileList[1][cl], cl_data_fileList[2][cl]]
        perm_ids = np.random.permutation(len(img_featList[0])).tolist()
        z_allList[0].append([np.squeeze(img_featList[0][perm_ids[i]]) for i in range(n_support+n_query)])     # stack each batch
        z_allList[1].append([np.squeeze(img_featList[1][perm_ids[i]]) for i in range(n_support+n_query)])
        z_allList[2].append([np.squeeze(img_featList[2][perm_ids[i]]) for i in range(n_support+n_query)])

    z_allList = [torch.from_numpy(np.array(z_allList[i])) for i in range(len(z_allList))]

    for model in modelList:
        model.n_query = n_query

    scoresList = []
    for i in range(len(modelList)):
        if adaptation:
            scores  = modelList[i].set_forward_adaptation(z_allList[i], is_feature = True)
        else:
            scores  = modelList[i].set_forward(z_allList[i], is_feature = True)
        scores = nn.Softmax(dim = 1)(scores)
        scoresList.append(scores)
    scores = reduce(lambda x,y: x + y, scoresList)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100
    return acc

if __name__ == '__main__':
    method_dict = {0:'baselinecutmix', 1:'Dropout', 2:'baseline_labelsmoothing'}
    params = parse_args('test')

    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)

    model1           = BaselineFinetune(model_dict[params.model], **few_shot_params)
    model2           = BaselineFinetune(model_dict[params.model], **few_shot_params)
    model3           = BaselineFinetune(model_dict[params.model], **few_shot_params)


    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'baselinecutmix' or params.method == 'baseline_labelsmoothing' or params.method == 'baselinemixup' or params.method == 'baselinemanifoldmixup':
        '''
        model1           = BaselineFinetune(model_dict[params.model], **few_shot_params)
        model2           = BaselineFinetune(model_dict[params.model], **few_shot_params)
        model3           = BaselineFinetune(model_dict[params.model], **few_shot_params)
        # model           = BaselineFinetune(model_dict[params.model], **few_shot_params)
        '''
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    models = [model1, model2, model3]
    models = [model.cuda() for model in models]
    # model = model.cuda()
    # checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    ckp_cutmix = './checkpoints/miniImagenet/Conv4_baselinecutmix_aug'
    ckp_dropout = './checkpoints/miniImagenet/Conv4Drop_baseline_aug'
    ckp_ls = './checkpoints/miniImagenet/Conv4_baseline_labelsmoothing_aug'
    checkpoint_dir_lst = [ckp_cutmix, ckp_dropout, ckp_ls]

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split

    checkpoint_dirs = checkpoint_dir_lst[:params.model_num]
    cl_data_file_list = []
    for ckp_dir in checkpoint_dirs:
        novel_file = os.path.join(ckp_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)
        
        cl_data_file_list.append(cl_data_file)
    for i in range(iter_num):
        acc = feature_evaluation_MultiModels(cl_data_file_list, models, n_query = 15, adaptation = params.adaptation, **few_shot_params)
        acc_all.append(acc)

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    with open('./record/results.txt' , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-adapted' if params.adaptation else ''
        if params.method in ['baseline', 'baseline++', 'baselinecutmix', 'baseline_labelsmoothing', 'baselinemixup', 'baselinemanifoldmixup']:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
        else:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.train_n_way, params.test_n_way)
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write('Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str))
