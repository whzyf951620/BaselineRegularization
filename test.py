import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import glob
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.EnsembleStrategy import Ensemble
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file
import pickle
from utils import to_one_hot

def get_errors(scores, y):
    y_one_hot = to_one_hot(y, 5)
    err = torch.abs(scores.data.cpu() - y_one_hot)
    err_sum = torch.sum(err).item()
    return err_sum

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
    err_sum = get_errors(scores, torch.LongTensor(y))
    return acc

def get_model(model_dict, params, few_shot_params):
    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'baselinecutmix' or params.method == 'baselinelabelsmoothing' or \
            params.method == 'baselinemixup' or params.method == 'baselinemanifoldmixup' or \
            params.method == 'baselinecutmixlabelsmoothing':
        model           = BaselineFinetune(model_dict[params.model], **few_shot_params)
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    elif params.method == 'baseline++labelsmoothing' or params.method == 'baseline++cutmix' or \
            params.method == 'baseline++mixup' or params.method == 'baseline++cutmixlabelsmoothing' or \
            params.method == 'baseline++manifoldmixup':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    else:
        raise ValueError('Unknown method')

    return model

def get_models(model_dict, method_list, model_list, few_shot_params):
    baselinelist = ['baseline', 'baselinecutmix', 'baselinelabelsmoothing', 'baselinecutmixlabelsmoothing']
    baselinepplist = ['baseline++', 'baseline++cutmix', 'baseline++labelsmoothing', 'baseline++cutmixlabelsmoothing']
    model_list_finetune = []
    for index, method in enumerate(method_list):
        if method in baselinelist:
            model = BaselineFinetune(model_dict[model_list[index]], **few_shot_params)
        else:
            model = BaselineFinetune(model_dict[model_list[index]], loss_type = 'dist', **few_shot_params)
        model_list_finetune.append(model)
    return model_list_finetune

def save_feats(path, feats, feats_item_list):
    for index, feat in enumerate(feats):
        epoch_path = os.path.join(path, 'epoch_' + str(index))
        if not os.path.exists(epoch_path):
            os.mkdir(epoch_path)
            
        mean_pkl_path = os.path.join(epoch_path, 'score_mean_' + str(index) + '.pkl')
        item_pkl_path = os.path.join(epoch_path, 'score_item_' + str(index) + '.pkl')
        with open(mean_pkl_path, 'wb') as f:
            pickle.dump(feat, f)
            f.close()
        with open(item_pkl_path, 'wb') as f_item:
            pickle.dump(feats_item_list[index], f_item)
            f_item.close()


if __name__ == '__main__':
    params = parse_args('test')

    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot, feat_aug = params.feat_aug)
    params.n_way = params.test_n_way
    params.n_support = params.n_shot
    params.n_query = 15

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    method_list = ['baseline', 'baseline++', 'baselinecutmix', 'baseline++cutmix' \
            'baselinelabelsmoothing', 'baselinemixup', 'baseline++mixup', 'baselinemanifoldmixup', \
            'baselinecutmixlabelsmoothing', 'baseline++cutmixlabelsmoothing', 'baseline++labelsmoothing']

    if not params.ensemble:
        model = get_model(model_dict, params, few_shot_params)
        model = model.cuda()

        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'

        split = params.split
        if params.save_iter != -1:
            split_str = split + "_" +str(params.save_iter)
        else:
            split_str = split

        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
    else:
        selected_method_list_flag = params.methodlist
        methods = selected_method_list_flag.split('_')
        method_list_regularization = []
        model_list = []
        for regularization in methods[1:]:
            if regularization == 'drop':
                model_list.append(params.model + 'Drop')
                method_list_regularization.append(methods[0])
                continue
            elif regularization == 'cutmix':
                model_list.append(params.model)
                method_list_regularization.append(methods[0] + regularization)
                continue
            elif regularization == 'labelsmoothing':
                model_list.append(params.model)
                method_list_regularization.append(methods[0] + regularization)
            else:
                raise ValueError('The methods have not be supported!')


        model_list_finetune = get_models(model_dict, method_list_regularization, model_list, few_shot_params)
        ckps_path = []
        for index, method in enumerate(method_list_regularization):
            checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, model_list[index], method)
            if params.train_aug:
                checkpoint_dir += '_aug'
            ckps_path.append(checkpoint_dir)
        split = params.split
        if params.save_iter != -1:
            split_str = split + "_" +str(params.save_iter)
        else:
            split_str = split

        ensemble = Ensemble(model_list_finetune, ckps_path, params, iter_num, params.ensemble_strategy)
        if params.clustering:
            acc_mean, acc_std, feats_item, feats_mean = ensemble.get_acc(n_query = 15)
            save_feats('clustering_analyses/miniImagenet/pkls_ensembel_meanscores', feats_mean, feats_item)
        else:
            acc_mean, acc_std = ensemble.get_acc(n_query = 15)

    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    with open('./record/results.txt' , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-adapted' if params.adaptation else ''
        exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )

