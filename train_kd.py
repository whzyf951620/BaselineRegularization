import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

from utils import classifier_teacher

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinetrain_cutmix import BaselineTrainCutMix
from methods.baselinetrain_labelsmoothing import BaselineTrainLS
from methods.baselinetrain_mixup import BaselineTrainMixUp
from methods.baselinetrain_knowledgedistillation import BaselineTrainKD
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file  

import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop( val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

def train_kd(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop_kd(epoch, base_loader,  optimizer)
        #, Fextractor_teacher = Fextractor_teacher, classifier_teacher = classifier_teacher) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir_kd):
            os.makedirs(params.checkpoint_dir_kd)

        acc = model.test_loop( val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir_kd, '{:d}.tar'.format(epoch))
            tmp = {}
            state_list = list(model.state_dict().keys())
            for key in state_list:
                if 'student' in key:
                    tmp[key] = model.state_dict()[key]
            # torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            torch.save({'epoch': epoch, 'state': tmp}, outfile)

    return model

def select_base_valfiles(params):

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
        val_file   = configs.data_dir['emnist'] + 'val.json' 
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json' 
        val_file   = configs.data_dir[params.dataset] + 'val.json' 

    return base_file, val_file

def select_image_size(params):

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    return image_size

def get_stop_epoch(params, method_list):
    if params.stop_epoch == -1: 
        if params.method in method_list:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400 # 400 original
            else:
                params.stop_epoch = 400 #default
        else: #meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600 #default

def get_baseline_methods(params):
    base_datamgr    = SimpleDataManager(image_size, batch_size = 512)
    base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
    val_datamgr     = SimpleDataManager(image_size, batch_size = 64)
    val_loader      = val_datamgr.get_data_loader( val_file, aug = False)
    
    if params.dataset == 'omniglot':
        assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
    if params.dataset == 'cross_char':
        assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

    if params.method == 'baseline':
        model           = BaselineTrain( model_dict[params.model], params.num_classes)
    elif params.method == 'baselinecutmix':
        model           = BaselineTrainCutMix(model_dict[params.model], params.num_classes, beta = 1., cutmix_prob = 1.)
    elif params.method == 'baselinecutmixlabelsmoothing':
        model           = BaselineTrainCutMixLS(model_dict[params.model], params.num_classes, beta = 1., cutmix_prob = 1.)
    elif params.method == 'baseline_labelsmoothing':
        model           = BaselineTrainLS(model_dict[params.model], params.num_classes, lam = 0.4)
    elif params.method == 'baselinemixup':
        model           = BaselineTrainMixUp(model_dict[params.model], params.num_classes, alpha = 1.)
    elif params.method == 'baseline++labelsmoothing':
        model           = BaselineTrainLS(model_dict[params.model], params.num_classes, lam = 0.4, loss_type = 'dist')
    elif params.method == 'baseline++cutmix':
         model           = BaselineTrainCutMix(model_dict[params.model], params.num_classes, beta = 1., cutmix_prob = 1., loss_type = 'dist')
    elif params.method == 'baseline++':
        model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')
    elif params.method == 'baseline++mixup':
        model           = BaselineTrainMixUp(model_dict[params.model], params.num_classes, loss_type = 'dist', alpha = 1.)
    elif params.method == 'baseline++cutmixlabelsmoothing':
        model           = BaselineTrainCutMixLS(model_dict[params.model], params.num_classes, loss_type = 'dist', beta = 1., cutmix_prob = 1.)

    return base_datamgr, base_loader, val_datamgr, val_loader, model

def get_checkpint_file(params):
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in method_list: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    ######loading the json for in-domain or cross-domain few-shot tasks
    base_file, val_file = select_base_valfiles(params)
    ######define the image size for different models and datasets
    image_size = select_image_size(params)
    method_list = ['baseline', 'baseline++', 'baseline++cutmix', 'baselinecutmix', \
            'baseline_labelsmoothing', 'baseline++labelsmoothing', 'baselinemixup', \
            'baseline++mixup', 'baselinecutmixlabelsmoothing', 'baseline++cutmixlabelsmoothing']
    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    optimization = 'Adam'
    get_stop_epoch(params, method_list) 

    if params.method in method_list:
        base_datamgr, base_loader, val_datamgr, val_loader, model = get_baseline_methods(params)
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    params.checkpoint_dir_kd = '%s/checkpoints_kd/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if params.train_aug:
        params.checkpoint_dir_kd += '_aug'
    if not params.method  in method_list:
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    KD = True
    # class_teacher = classifier_teacher()
    if params.model == 'Conv4':
        Fextractor_teacher = backbone.Conv4
    else:
        raise NotImplementedError('Use Conv4')
    if KD:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'
        modelfile   = get_best_file(checkpoint_dir)
        model = BaselineTrainKD(Fextractor_teacher, model_dict[params.model], params.num_classes, 0.9, 20).cuda()
        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        tmp_classifier = {}
        for i, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            elif "classifier." in key:
                newkey = key.replace("classifier.","")
                tmp_classifier[newkey] = state.pop(key)

        model.feature.load_state_dict(state)
        model.classifier.load_state_dict(tmp_classifier)
        model = train_kd(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)
