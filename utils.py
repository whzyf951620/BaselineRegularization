import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def find_sen(sigma, test_loader, clf, device, num_classes):
    with torch.no_grad():
        clf.eval()

        errors = []
        outs = []

        for data, target in test_loader:
            data = Variable(data).cuda()
            target = Variable(target).cuda()
            noise = Variable(data.new(data.size()).normal_(0, sigma)).cuda()

            data_noisy = data + noise

            output = clf(data)
            output_noisy = clf(data_noisy)

            error = output_noisy - output
            error = error.cpu().detach().numpy()

            for ind in np.arange(np.shape(target.data.tolist())[0]):
                temp = 0

                for i in range(num_classes):
                    temp += error[ind][i]

                errors.append(temp/num_classes)

        errors = np.asarray(errors)
        return np.var(errors)


###############schmidt orthogonalization##############
def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)

###############manifold mixup####################
def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    target_reweighted = target_reweighted.cuda()
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return out, target_reweighted

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
 
def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.shape[0], num_classes)
    y_onehot.zero_()
    inp_exp = inp.unsqueeze(1).data.cpu()
    y_onehot.scatter_(1, inp_exp, 1)
    #return Variable(y_onehot.cuda(),requires_grad=False)
    return y_onehot

def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.: 
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam 

###############loss_fn_kd#################
# The method is copy from https://github.com/peterliht/knowledge-distillation-pytorch
class loss_fn_kd(object):
    def __init__(self, params_dict):
        super(loss_fn_kd).__init__()
        self.alpha = params_dict['alpha']
        self.T = params_dict['temperature']

    def __call__(self, outputs, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/self.T, dim=1),\
                                 F.softmax(teacher_outputs/self.T, dim=1)) * \
                                 (self.alpha * self.T * self.T) + \
                                 F.cross_entropy(outputs, labels) * (1. - self.alpha)
        return KD_loss

    def __repr__(self):
        return 'Knowledge Distillation of alpha={} and temperature={}'.format(str(self.alpha), str(self.temperature))

###############Mixup####################
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x +(1-lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

class mixup_criterion(object):
    def __init__():
        super(mixup_criterion, self).__init__()

    def __call__(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def __repr__():
        return "criterion of MixUp"

###############label smoothing#################
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, lam):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.lam = lam

    def forward(self, x, y):
        confidence = 1. - self.lam
        logprobs = F.log_softmax(x, dim = -1)
        nll_loss = -logprobs.gather(dim = -1, index = y.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim = -1)
        loss = confidence * nll_loss + self.lam * smooth_loss
        return loss.mean()

###############for cutmix#################
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
################for mixup##################
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
#################original####################
def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl]])) 

    return np.mean(cl_sparsity)

class classifier_teacher(nn.Module):
    def __init__(self, in_channels = 1600, num_class = 64):
        super(classifier_teacher, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.classifier = nn.Linear(self.in_channels, self.out_channels)
        self.classifier.bias.data.fill_(0)

    def forward(self, x):
        return self.classifier(x)


class SoftNearestNeighborLoss(nn.Module):
    def __init__(self, samples, labels, T = None):
        super(SoftNearestNeighborLoss, self).__init__()
        self.samples = samples
        self.labels = labels
        self.Temperature = torch.nn.Parameter(torch.tensor(float(-10))).cuda()

    def forward(self):
        loss = 0
        bs = self.samples.shape[0]
        self.samples = self.samples.view(bs, -1)
        for index, sample in enumerate(self.samples):
            sample = sample.repeat(bs, 1)
            dists = torch.pow(sample - self.samples, 2)
            diff = torch.exp(-dists / self.Temperature)
            label_cur = self.labels[index]
            sameClassDists = diff[self.labels == label_cur]
            loss += torch.log(sameClassDists.sum() / (diff.sum() + 1e-6))

        return -loss / bs
