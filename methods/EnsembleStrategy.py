import torch
import torch.nn as nn
import numpy as np
import data.feature_loader as feat_loader
from utils import to_one_hot
from collections import Counter
import os
import random

def get_errors(scores, y):
    y_one_hot = to_one_hot(y, 5)
    err = torch.abs(scores.data.cpu() - y_one_hot)
    err_sum = torch.sum(err).item()
    return err_sum

def list_add(feats):
    f0, f1 = feats
    assert len(f0) == len(f1)
    for index in range(len(f0)):
        f0[index] += f1[index]

    return f0

def feats_add(feats, add_type = 'sum'):
    for ft in feats:
        ft = np.array(ft)
        ft = np.expand_dims(ft, axis = 0)

    feats_np = np.concatenate(feats, axis = 0)
    if add_type == 'sum':
        feats_sum = np.sum(feats_np, axis = 0)
        return feats_sum
    else:
        feats_mean = np.mean(feats_np, axis = 0)
        return feats_sum

def classifier_training(z_all, model, n_query, adaptation = False):
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    scores = nn.Softmax(dim = 1)(scores)
    return scores, model

def accuracy(pred, n_way, n_query):
    # pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    if len(pred.shape) != len(y.shape):
        pred = pred.reshape(-1)
    acc = np.mean(pred == y)*100
    return acc

def select_feature_tasks_diff(cl_data_file, n_way = 5, n_support = 5, n_query = 15, select_class = None):
    class_list = cl_data_file.keys()
    # select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query)])     # stack each batch
    z_all = torch.from_numpy(np.array(z_all))
    return z_all

def select_feature_tasks_same(cl_data_file, n_way = 5, n_support = 5, n_query = 15, select_class = None, img_indices_list = None):
    class_list = cl_data_file.keys()
    # select_class = random.sample(class_list,n_way)
    z_all  = []
    for index, cl in enumerate(select_class):
        img_feat = cl_data_file[cl]
        img_indices = img_indices_list[index]
        z_all.append([np.squeeze(img_feat[i]) for i in img_indices])# stack each batch
    z_all = torch.from_numpy(np.array(z_all))
    return z_all

def predict(scores):
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    return pred

def voting(preds):
    for index, ft in enumerate(preds):
        ft = np.expand_dims(ft, axis = 1)
        preds[index] = ft
    feats_np = np.concatenate(preds, axis = 1)

    pred_final = np.zeros(preds[0].shape)
    for index, feat in enumerate(feats_np):
        voting_vector = np.bincount(feat)
        decision = np.argmax(voting_vector)
        pred_final[index] = decision

    return pred_final

class Ensemble(object):
    def __init__(self, models, ckps_path, params, iter_num = 600, strategy = 'VoteScores', split_str='novel'):
        super(Ensemble, self).__init__()
        assert strategy in ['MultiFeats', 'MultiClassifiers', 'MeanScores', 'VoteScores', 'FeatAugmentation']
        self.models = models
        self.ckps_path = ckps_path
        self.params = params
        self.strategy = strategy
        self.split_str = split_str
        self.cl_data_file_list = []
        self.get_cl_data_file_list()
        self.iter_num = iter_num

    def get_cl_data_file_list(self):
        for ckp_dir in self.ckps_path:
            novel_file = os.path.join(ckp_dir.replace("checkpoints","ensemble_features"), self.split_str + ".hdf5") #defaut split = novel, but you can also test base or val classes
            cl_data_file = feat_loader.init_loader(novel_file)

            self.cl_data_file_list.append(cl_data_file)

    def get_acc(self, n_query):
        acc_all = []
        feats = []
        feats_item_list = []
        for i in range(self.iter_num):
            featurelist = []
            img_indices_list = []
            select_class = random.sample(self.cl_data_file_list[0].keys(), self.params.n_way)
            for i in range(self.params.n_way):
                img_indices = random.sample([j for j in range(600)], self.params.n_support + n_query)
                img_indices_list.append(img_indices)
            for cl_data_file in self.cl_data_file_list:
                features = select_feature_tasks_same(cl_data_file, self.params.n_way, self.params.n_support, n_query, select_class, img_indices_list)
                featurelist.append(features)
            if self.strategy == 'MultiFeats':
               acc = self.multifeatstrategy(featurelist)
            
            elif self.strategy == 'MultiClassifiers':
               acc = self.multiclassifierstrategy(featurelist)

            elif self.strategy == 'MeanScores':
                if self.params.clustering:
                    acc, feats_item, feats_mean = self.meanscorestrategy(featurelist)
                    feats.append(feats_mean)
                    feats_item_list.append(feats_item)
                else:
                    acc = self.meanscorestrategy(featurelist)
            elif self.strategy == 'FeatAugmentation':
                acc = self.feataugmentation(featurelist)
            else:
                acc = self.votescorestrategy(featurelist)
            acc_all.append(acc)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.params.clustering:
            return acc_mean, acc_std, feats_item_list, feats
        else:
            return acc_mean, acc_std

    def multifeatstrategy(self, featurelist):
        features = feats_add(featurelist, add_type = 'mean')
        scores, model = classifier_training(features, self.models[0], self.params.n_query, self.params.adaptation)
        pred = predict(scores)
        acc = accuracy(pred, self.params.n_way, self.params.n_query)
        return acc

    def multiclassifierstrategy(self, featurelist):
        modelist = []
        for index, feature in enumerate(featurelist):
            scores, model = classifier_training(feature, self.models[index], self.params.n_query, self.params.adaptation)
            modelist.append(model)

    def meanscorestrategy(self, featurelist):
        scoreslist = []
        for index, feature in enumerate(featurelist):
            scores, _ = classifier_training(feature, self.models[index], self.params.n_query, self.params.adaptation)
            scoreslist.append(scores)

        scores_mean = torch.mean(torch.stack(scoreslist, dim = 0), dim = 0)
        # scores_mean = nn.Softmax(dim = 1)(scores_mean)
        pred = predict(scores_mean)
        acc = accuracy(pred, self.params.n_way, self.params.n_query)
        # y = torch.LongTensor(np.repeat(range(self.params.n_way), self.params.n_query))
        # err_sum = get_errors(scores_mean, y)
        # print(err_sum)
        print(acc)
        if self.params.clustering:
            return acc, scoreslist, scores_mean
        else:
            return acc

    def feataugmentation(self, featureslist):
        feats_support = [feats[:, :self.params.n_support, :] for feats in featureslist]
        feats_support = torch.cat(feats_support, dim = 1)
        feats_query = [feats[:, self.params.n_support:, :] for feats in featureslist]
        feats_query = torch.cat(feats_query, dim = 1)
        feats = torch.cat([feats_support, feats_query], dim = 1)
        scores, _ = classifier_training(feats, self.models[0], self.params.n_query, self.params.adaptation)
        scores = scores.view(self.params.n_query * self.params.n_way, -1, self.params.n_way)
        scores = torch.mean(scores, dim=1)
        pred = predict(scores)
        acc = accuracy(pred, self.params.n_way, self.params.n_query)
        print(acc)
        return acc

    def votescorestrategy(self, featurelist):
        scoreslist = []
        for index, feature in enumerate(featurelist):
            scores, _ = classifier_training(feature, self.models[index], self.params.n_query, self.params.adaptation)
            scoreslist.append(scores)

        predlist = []
        for scores in scoreslist:
            pred = predict(scores)
            predlist.append(pred)

        pred = voting(predlist)
        acc = accuracy(pred, self.params.n_way, self.params.n_query)
        return acc
