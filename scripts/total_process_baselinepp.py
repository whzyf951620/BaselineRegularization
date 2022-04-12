import os

# method_list = ['baseline++', 'baseline++cutmix', 'baseline++labelsmoothing', 'baseline++mixup', 'baseline++cutmixlabelsmoothing']
method_list = ['baseline++cutmixlabelsmoothing']

for method in method_list:
    save_path = os.path.join('./record', 'Conv4' + method + '.txt')
    os.system('CUDA_VISIBLE_DEVICES=1 python train.py --model Conv4 --method ' + method + ' --dataset miniImagenet --train_aug')
    os.system('CUDA_VISIBLE_DEVICES=1 python save_features.py --model Conv4 --method ' + method + ' --dataset miniImagenet --train_aug')
    os.system('CUDA_VISIBLE_DEVICES=1 python test.py --model Conv4 --method ' + method + ' --dataset miniImagenet --train_aug > ' + save_path + ' 2>&1')
